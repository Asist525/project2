from __future__ import annotations
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import numpy as np
import matplotlib.pyplot as plt

from datetime import timedelta
import logging
logger = logging.getLogger(__name__)
import random
KR_US_BDAY = CustomBusinessDay(calendar="KRX")  # 필요 시 NYSE 일정 병합
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from nomalization import *

from tabulate import tabulate
import time

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_DIR)

# Django 설정 파일 로드
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()

# 모델 임포트
from ai.models import AI_REWARD
import math

def partial_fit_batches(
        batches,
        train_months: int = 3,
        pred_months: int = 1,
        model: SGDRegressor | None = None,
        scaler: MinMaxScaler | None = None,
        reward_weight: float = 0.1,
):
    # ────────────────────────────────────────
    # 0) 초기 객체 준비 ───────────────────────
    # ────────────────────────────────────────
    fit_cols = ["Open", "High", "Low", "Volume", "Prev_Close"]   ### MOD: Prev_Close 포함
    target_col = "Close"

    if model is None:
        model = SGDRegressor(
            loss="huber",
            epsilon=1.35,
            random_state=42,
            penalty="l2",
            alpha=1e-4,                 # ← 고정 α
            learning_rate="constant",   # ← 고정 학습률
            eta0=0.03,                  # ← 초기 학습률
            fit_intercept=True,
        )
    if scaler is None:                                            ### MOD: 첫 호출 때 스케일러 생성
        scaler = MinMaxScaler()

    # 모델에 feature 이름 심어두기(최초 1회)
    if not hasattr(model, "feature_names_in_"):
        model.feature_names_in_ = np.array(fit_cols)

    results = []

    # ────────────────────────────────────────
    # 1) 배치 루프 ───────────────────────────
    # ────────────────────────────────────────
    for train_b, pred_b in walk_forward_batches(batches, train_months, pred_months):
        train_b = train_b.copy()

        # 1-a) Prev_Close 생성
        train_b["Prev_Close"] = train_b[target_col].shift(1).bfill()

        # 1-b) 스케일러 업데이트(점진 / 한 번만!)
        scaler.partial_fit(train_b[fit_cols])

        # 1-c) 특성 행렬 / 타깃 벡터
        X_train = pd.DataFrame(
            scaler.transform(train_b[fit_cols]),                   ### MOD: 바로 DataFrame 반환
            columns=fit_cols,
            index=train_b.index,
        )
        y_train = train_b[target_col].values

        # 1-d) **중복 호출 제거**  (한 배치당 한 번만 학습)
        model.partial_fit(X_train, y_train)                        ### MOD: 두 번째 partial_fit 삭제

        # ───────────────────────────────────
        # 2) 다중 스텝 예측 ──────────────────
        # ───────────────────────────────────
        last_row_scaled = X_train.iloc[-1].values.reshape(1, -1)
        idx_prev = fit_cols.index("Prev_Close")

        future_dates = (
            pred_b["Date"].dt.strftime("%Y-%m-%d").values
            if pred_b is not None
            else pd.date_range(
                start=train_b["Date"].max() + pd.Timedelta(days=1),
                periods=pred_months * 21,  # 영업일 근사
                freq="B",
            ).strftime("%Y-%m-%d")
        )

        # ─ 2) 다중 스텝 예측 (학습 단계) ─────────────────
        last_row_scaled = X_train.iloc[-1].values.reshape(1, -1)
        idx_prev = fit_cols.index("Prev_Close")

        y_pred = []
        for _ in range(len(pred_b)):
            # 2-a. 예측
            next_scaled = model.predict(
                pd.DataFrame(last_row_scaled, columns=fit_cols)
            )[0]
            y_pred.append(next_scaled)

            # 2-b. 전체 피처를 학습 시처럼 업데이트
            # → 단순히 Prev_Close 만 갱신하면 다시 고정점
            last_row = last_row_scaled[0].copy()

            # **Prev_Close는 이전 예측값으로**
            last_row[idx_prev] = next_scaled
            
            # **나머지 피처는 약간의 임의성 추가 (더 강하게)**
            feature_noise = np.random.normal(0, 0.2, size=idx_prev)   # 노이즈 크기 증가
            dropout_mask = np.random.rand(idx_prev) > 0.3         # 일부 피처는 그대로 유지
            last_row[:idx_prev] += feature_noise * dropout_mask

            last_row_scaled = last_row.reshape(1, -1)



        # ───────────────────────────────────
        # 3) 평가 지표 계산 ───────────────────
        # ───────────────────────────────────
        if pred_b.empty:
            continue

        y_true = pred_b[target_col].values
        if any(pd.isna(y_true)) or len(y_true) == 0:
            continue

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = (
            np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100
        )
        r2 = r2_score(y_true, y_pred)

        # 3-b) 규제 강도(α) 보상
        reward = 1 / (1 + mape)
        current_alpha = model.alpha
        if not np.isfinite(current_alpha) or current_alpha < 1e-4:
            current_alpha = 1e-4  # ← 초기값으로 리셋
            
        if not np.isfinite(reward) or reward < 0.01:
            reward = 0.01
        new_alpha_candidate = current_alpha * (1 - reward_weight * reward)
        new_alpha = max(new_alpha_candidate, 1e-8)   # nan 방지 & 하한

        model.alpha = new_alpha

        # ───────────────────────────────────
        # 4) 결과 저장 ───────────────────────
        # ───────────────────────────────────
        results.append(
            {
                "train_period": (
                    train_b["Date"].min().strftime("%Y-%m-%d"),
                    train_b["Date"].max().strftime("%Y-%m-%d"),
                ),
                "pred_period": (
                    pred_b["Date"].min().strftime("%Y-%m-%d"),
                    pred_b["Date"].max().strftime("%Y-%m-%d"),
                ),
                "mape": mape,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "reward": reward,
                "y_pred": y_pred,
                "y_true": y_true,
            }
            
        )
        
    # 문제 생기면 디버깅할것 a가 NAN이면 안됨
    # print("α =", model.alpha)
    # print("t(step) =", model.t_)                  # 반복 회수
    # print("coef_ =", model.coef_)                 # 각 피처 가중치
    # print("intercept_ =", model.intercept_)
    # print("‖w‖₂ =", np.linalg.norm(model.coef_))
    # print(model.alpha, model.coef_, model.intercept_)
    # print("mape =", mape, "reward =", reward)
    # print("model.alpha BEFORE update =", model.alpha)


    # 실제값과 예측값 출력
    comparison_df = pd.DataFrame({
        "실제값 (y_true)": y_true,
        "예측값 (y_pred)": y_pred
    })
    print("실제값 vs 예측값 (일부 샘플)")
    print(tabulate(comparison_df.head(10), headers="keys", tablefmt="fancy_grid"))

    # 성능 지표 출력
    print("배치 성능 요약")
    performance_data = [
        ["MAPE (%)", f"{mape:.2f}"],
        ["MAE", f"{mae:.4f}"],
        ["RMSE", f"{rmse:.4f}"],
        ["R2", f"{r2:.4f}"],
        ["Reward", f"{reward:.4f}"],
    ]
    print(tabulate(performance_data, headers=["Metric", "Value"], tablefmt="fancy_grid"))

    save_data(mape,mae,rmse,r2,reward)

    return model, scaler


def save_data(mape, mae, rmse, r2, reward):
    if math.isnan(mape) or math.isnan(mae) or math.isnan(rmse) or math.isnan(r2) or math.isnan(reward):
        print("❌ 저장 실패: nan 값이 포함되어 있습니다.")
        return
    # AI_REWARD 모델 인스턴스 생성
    reward_instance = AI_REWARD(
        MAPE=mape,
        MAE=mae,
        RMSE=rmse,
        R2=r2,
        REWARD=reward,
    )
    
    # 단일 인스턴스는 bulk_create가 아닌 save()를 사용해야 함
    reward_instance.save()


def main():
    ticker = "QQQ"
    shuffled_batches = mini_batch_normalization(ticker)

    while(True):
        # shuffled_batches = mini_batch_normalization(ticker) # 새로운 데이터를 삽입해야할 때 사용
        try:
            model = load_model("models/price_model.pkl")
        except FileNotFoundError:
            model = SGDRegressor(loss='huber', epsilon=1.35, random_state=42)

        # 2. 스케일러 로드 (없으면 새로 생성)
        try:
            scaler = load_or_init_scaler("models/price_scaler.pkl",fit_cols)
        except FileNotFoundError:
            scaler = MinMaxScaler()

        # 3. 모델 학습
        model, scaler = partial_fit_batches(shuffled_batches, model=model, scaler=scaler)

        # 4. 모델 저장
        save_model(model, "models/price_model.pkl")
        save_scaler(scaler, "models/price_scaler.pkl")

        # ** 모델과 스케일러 로드 **
        model = load_model("models/price_model.pkl")
        scaler = load_scaler("models/price_scaler.pkl")


        # =============== 디버깅 =====================
        # 모델 통계 출력
        # if model is not None:
        #     weights_rounded = [float(f"{w:.2f}") for w in model.coef_]
        #     intercept_rounded = float(f"{model.intercept_[0]:.2f}")
        #     print("\n모델 가중치:", weights_rounded)
        #     print("모델 절편:", intercept_rounded)

        time.sleep(3)
if __name__ == "__main__":
    main()
