# -*- coding: utf-8 -*-
"""
batch 학습 + 예측 + 모델/스케일러 저장
리팩터링: 다중 지표 보상, 안정적 α 업데이트, 성능·가독성 개선
"""
from __future__ import annotations

# ─────────────────────  표준 라이브러리  ─────────────────────
import os, sys, time, math, logging, random
from datetime import timedelta
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Sequence

# ─────────────────────  외부 라이브러리  ─────────────────────
import numpy as np
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from tabulate import tabulate

# ─────────────────────  시각화 (선택)  ─────────────────────
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

# ─────────────────────  Django ORM  ─────────────────────
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_DIR)
import django, warnings
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()
from ai.models import AI_REWARD

# ─────────────────────  유틸 / 내부 모듈  ─────────────────────
from nomalization import (          # 기존 모듈 경로 유지
    walk_forward_batches,
    mini_batch_normalization,
    load_model, save_model,
    load_scaler, save_scaler,
    load_or_init_scaler,
)

# ─────────────────────  상수·전역  ─────────────────────
logging.basicConfig(level=logging.INFO)
LOGGER                  = logging.getLogger(__name__)
EPS                     = 1e-8
ALPHA_MIN               = 1e-8
ALPHA_RESET             = 1e-4
DECAY_K                 = 0.5
DEFAULT_REWARD_WEIGHT   = (0.4, 0.2, 0.2, 0.2)   # MAPE·MAE·RMSE·R2
FIT_COLS: Sequence[str] = ["Open", "High", "Low", "Volume", "Prev_Close"]
TARGET_COL              = "Close"
KR_US_BDAY              = CustomBusinessDay(calendar="KRX")  # 필요 시 KRX·NYSE 병합

# ─────────────────────  보상·α 헬퍼  ─────────────────────
def _composite_reward(
    mape: float, mae: float, rmse: float, r2: float,
    w: Tuple[float, float, float, float] = DEFAULT_REWARD_WEIGHT,
) -> float:
    """MAPE·MAE·RMSE·R² → 0~1 스케일 보상 후 가중 평균"""
    f      = lambda x: 1.0 / (1.0 + x)           # 작을수록 ↑
    f_r2   = lambda x: max((x + 1) / 2, 0.0)     # (-1~1) → (0~1)
    reward = (
        w[0] * f(mape) +
        w[1] * f(mae)  +
        w[2] * f(rmse) +
        w[3] * f_r2(r2)
    ) / sum(w)
    return max(reward, 1e-2)                     # 하한

def _update_alpha(alpha: float, reward: float) -> float:
    """exp 감쇠로 규제 강도 업데이트"""
    new_alpha = alpha * np.exp(-DECAY_K * reward)
    return max(new_alpha, ALPHA_MIN)

# ─────────────────────  결과 구조체  ─────────────────────
@dataclass
class BatchResult:
    train_period: Tuple[str, str]
    pred_period: Tuple[str, str]
    mape: float
    mae: float
    rmse: float
    r2: float
    reward: float
    y_true: np.ndarray
    y_pred: np.ndarray

# ─────────────────────  핵심 함수  ─────────────────────
def partial_fit_batches(
    batches: Iterable[pd.DataFrame],
    train_months: int = 3,
    pred_months: int = 1,
    model: SGDRegressor | None = None,
    scaler: MinMaxScaler | None = None,
    reward_weights: Tuple[float, float, float, float] = DEFAULT_REWARD_WEIGHT,
    verbose: bool = True,
) -> Tuple[SGDRegressor, MinMaxScaler, List[BatchResult]]:
    """walk-forward 배치 학습 + 다중 스텝 예측"""

    # ── 모델·스케일러 준비 ───────────────────
    if model is None:
        model = SGDRegressor(
            loss="huber",
            epsilon=1.35,
            random_state=42,
            penalty="l2",
            alpha=ALPHA_RESET,
            learning_rate="constant",
            eta0=0.03,
            fit_intercept=True,
        )
    if scaler is None:
        scaler = MinMaxScaler()

    if not hasattr(model, "feature_names_in_"):
        model.feature_names_in_ = np.array(FIT_COLS)

    results: List[BatchResult] = []

    # ── 배치 루프 ───────────────────────────
    for train_b, pred_b in walk_forward_batches(batches, train_months, pred_months):
        if train_b.empty or pred_b.empty:
            continue

        # Prev_Close 추가
        train_b = train_b.copy()
        train_b["Prev_Close"] = train_b[TARGET_COL].shift(1).bfill()

        # 스케일러 점진 학습 & 변환
        scaler.partial_fit(train_b[FIT_COLS])
        X_train = pd.DataFrame(
            scaler.transform(train_b[FIT_COLS]),
            columns=FIT_COLS,
            index=train_b.index,
        )
        y_train = train_b[TARGET_COL].values

        model.partial_fit(X_train, y_train)

        # ── 다중 스텝 예측 ─────────────────────────
        last_row = X_train.iloc[-1].copy()   # ← DataFrame에서 마지막 행 복사
        idx_prev = FIT_COLS.index("Prev_Close")

        y_pred: List[float] = []
        for _ in range(len(pred_b)):
            # DataFrame 형태로 예측
            next_val = float(model.predict(
                pd.DataFrame([last_row], columns=FIT_COLS)
            )[0])
            y_pred.append(next_val)

            # **Prev_Close 갱신 (위치 기반)**
            last_row.iloc[idx_prev] = next_val  # ← 여기서 iloc 사용

            # **나머지 피처에 노이즈 추가**
            noise = np.random.normal(0.0, 0.15, size=idx_prev)
            keep_mask = np.random.rand(idx_prev) > 0.3
            last_row.iloc[:idx_prev] += noise * keep_mask

        # ── 다중 스텝 예측 후 평가 지표 계산 ─────────────────────────
        y_true = np.asarray(pred_b[TARGET_COL].values, dtype=float)  # ← 명시적 float 변환

        # NaN 검사
        if pd.isna(y_true).any() or len(y_true) == 0:
            LOGGER.warning("❌ y_true에 NaN 또는 빈 배열이 포함되어 있음 → 건너뜀")
            continue

        # ── 지표 계산 ───────────────────────
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = (
            np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + EPS))) * 100.0
        )
        r2   = r2_score(y_true, y_pred)

        reward = _composite_reward(mape, mae, rmse, r2, reward_weights)
        model.alpha = _update_alpha(
            model.alpha if np.isfinite(model.alpha) else ALPHA_RESET,
            reward,
        )

        # ── 결과 저장 ───────────────────────
        results.append(
            BatchResult(
                train_period=(
                    train_b["Date"].min().strftime("%Y-%m-%d"),
                    train_b["Date"].max().strftime("%Y-%m-%d"),
                ),
                pred_period=(
                    pred_b["Date"].min().strftime("%Y-%m-%d"),
                    pred_b["Date"].max().strftime("%Y-%m-%d"),
                ),
                mape=mape,
                mae=mae,
                rmse=rmse,
                r2=r2,
                reward=reward,
                y_true=y_true,
                y_pred=np.array(y_pred),
            )
        )

        # ── 로그 & 출력 ────────────────────
        if verbose:
            perf_table = [
                ["MAPE (%)", f"{mape:.0f}"],          # ← 소수점 없이 정수형
                ["MAE",      f"{mae:.4f}"],
                ["RMSE",     f"{rmse:.4f}"],
                ["R2",       f"{r2:.4f}"],
                ["Reward",   f"{reward:.4f}"],
                ["α",        f"{model.alpha:.8f}"],   # ← 8자리 고정 소수점
            ]
            print(tabulate(perf_table, headers=["Metric", "Value"],
                        tablefmt="fancy_grid"))

        # db 저장
        save_data(mape, mae, rmse, r2, reward)

    return model, scaler, results

# ─────────────────────  DB 저장  ─────────────────────
def save_data(mape: float, mae: float, rmse: float, r2: float, reward: float):
    # 타입 변환 (Decimal → float)
    values = np.array([mape, mae, rmse, r2, reward], dtype=float)

    # NaN 검사
    if np.isnan(values).any():
        LOGGER.warning("❌ nan 포함 → DB 저장 건너뜀")
        return

    # 범위 제한
    mape = min(mape, 9999.999999)
    mae = min(mae, 9999.999999)
    rmse = min(rmse, 9999.999999)
    r2 = min(max(r2, -9999.999999), 9999.999999)
    reward = min(reward, 9999.999999)

    # DB 저장
    AI_REWARD.objects.create(
        MAPE=mape,
        MAE=mae,
        RMSE=rmse,
        R2=r2,
        REWARD=reward,
    )


# ─────────────────────  메인 루프  ─────────────────────
def main():
    ticker = "QQQ"
    shuffled_batches = mini_batch_normalization(ticker)

    while True:
        # 1) 모델·스케일러 로드 (없으면 새로)
        try:
            model = load_model("models/price_model.pkl")
        except FileNotFoundError:
            model = None

        try:
            scaler = load_or_init_scaler("models/price_scaler.pkl", FIT_COLS)
        except FileNotFoundError:
            scaler = None

        # 2) 학습
        model, scaler, _ = partial_fit_batches(
            shuffled_batches, model=model, scaler=scaler, verbose=True
        )

        # 3) 저장
        save_model(model,  "models/price_model.pkl")
        save_scaler(scaler, "models/price_scaler.pkl")

        #time.sleep(3)

# ─────────────────────  엔트리  ─────────────────────
if __name__ == "__main__":
    main()
