# -*- coding: utf-8 -*-
"""
Batch 학습 + 예측 + 모델/스케일러 저장 (Refactored)
- 데이터 누수 방지 (Prev_Close)
- 온라인 러닝 안전성 강화 (alpha 갱신 방식 수정)
- 지표별 보상 개선 + bulk DB 저장
- 가독성·테스트 재현성·I/O 최적화
"""


# mape 주로하는 단기 주가 예측 모델
# 단기로 사용할 경우 예측을 21일이 아닌 7일 미만으로 해야 의미있는 데이터를 얻을 수 있음


from __future__ import annotations

# ─────────────────────  표준 라이브러리  ─────────────────────
import os, sys, time, logging
from datetime import timedelta
from dataclasses import dataclass
from typing import Iterable, List, Tuple, Sequence

# ─────────────────────  외부 라이브러리  ─────────────────────
import numpy as np
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tabulate import tabulate

# ─────────────────────  Django ORM bootstrap  ─────────────────────
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_DIR)
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()
from ai.models import AI_REWARD

# ─────────────────────  내부 모듈  ─────────────────────
from nomalization import (
    walk_forward_batches,
    mini_batch_normalization,
    load_model, save_model,
    load_scaler, save_scaler,
    load_or_init_scaler,
    add_sma_features
)

# ─────────────────────  상수·전역  ─────────────────────
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

np.random.seed(42)  # 재현성 확보 (운영 시 삭제)

EPS                     = 1e-8
ALPHA_MIN               = 1e-8
ALPHA_RESET             = 1e-4
DECAY_K                 = 0.5
DEFAULT_REWARD_WEIGHT   = (0.4, 0.2, 0.2, 0.2)   # MAPE·MAE·RMSE·R2
EXTRA_SMA = [f"SMA_{p}" for p in (5, 20, 60)]
FIT_COLS   = ["Open", "High", "Low", "Volume", "Prev_Close"] + EXTRA_SMA
TARGET_COL              = "Close"
KR_US_BDAY              = CustomBusinessDay(calendar="KRX")  # 필요 시 KRX·NYSE 병합

# ─────────────────────  보상·α 헬퍼  ─────────────────────

def _composite_reward(
    mape: float, mae: float, rmse: float, r2: float,
    w: Tuple[float, float, float, float] = DEFAULT_REWARD_WEIGHT,
) -> float:
    """MAPE·MAE·RMSE·R² → 0~1 스케일 보상 후 가중 평균"""
    f_error = lambda x: 1.0 / (1.0 + x)           # 작을수록 ↑
    f_r2    = lambda x: 1.0 / (1.0 + np.exp(-5 * x))  # S-shaped, x ∈ (−∞,1]
    reward  = (
        w[0] * f_error(mape) +
        w[1] * f_error(mae)  +
        w[2] * f_error(rmse) +
        w[3] * f_r2(r2)
    ) / sum(w)
    return reward  # 0~1 사이


def _update_alpha(cur_alpha: float, reward: float) -> float:
    """exp 감쇠로 규제 강도 업데이트, ALPHA_MIN 이하로는 잘라냄"""
    new_alpha = cur_alpha * np.exp(-DECAY_K * reward)
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
    batches: Iterable[Tuple[pd.DataFrame, pd.DataFrame]],
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
            learning_rate="invscaling",
            eta0=0.01,
            power_t=0.25,
            fit_intercept=True,
        )
    if scaler is None:
        scaler = MinMaxScaler()

    results: List[BatchResult] = []

    # ── 배치 루프 ───────────────────────────
    for train_b, pred_b in batches:
        if train_b.empty or pred_b.empty:
            continue
        if "SMA_5" not in train_b.columns:
            train_b = add_sma_features(train_b, periods=(5,20,60))
            pred_b  = add_sma_features(pred_b,  periods=(5,20,60))

        # Prev_Close (누수 방지) + 첫 행 제거
        train_b = train_b.copy()
        train_b["Prev_Close"] = train_b[TARGET_COL].shift(1)
        train_b.dropna(subset=["Prev_Close"], inplace=True)  # 첫 행 제거

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
        last_row = X_train.iloc[-1].copy()
        idx_prev = FIT_COLS.index("Prev_Close")

        y_pred: List[float] = []
        for _ in range(len(pred_b)):
            next_val = float(model.predict(pd.DataFrame([last_row], columns=FIT_COLS))[0])
            y_pred.append(next_val)

            # Prev_Close 갱신
            last_row.iloc[idx_prev] = next_val

            # 탐색 노이즈 (µ=0, σ=0.01) – Open~Volume 피처에만
            noise = np.random.normal(0.0, 0.01, size=idx_prev)
            last_row.iloc[:idx_prev] += noise

        # ── 지표 계산 ───────────────────────
        y_true = pred_b[TARGET_COL].astype(float).values
        y_pred = np.array(y_pred)

        # NaN 필터링
        valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if not valid_mask.any():
            LOGGER.warning("❌ No valid samples (all NaN) → Batch skipped")
            continue
        y_true = y_true[valid_mask]
        y_pred = y_pred[valid_mask]

        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + EPS))) * 100.0
        if mape > 100:
            continue
        r2   = r2_score(y_true, y_pred)

        reward = _composite_reward(mape, mae, rmse, r2, reward_weights)

        # alpha 갱신 (다음 배치부터 반영)
        new_alpha = _update_alpha(model.alpha if np.isfinite(model.alpha) else ALPHA_RESET, reward)
        model.set_params(alpha=new_alpha)

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
            table = [
                ["MAPE (%)", f"{mape:.0f}"],
                ["MAE",      f"{mae:.4f}"],
                ["RMSE",     f"{rmse:.4f}"],
                ["R2",       f"{r2:.4f}"],
                ["Reward",   f"{reward:.4f}"],
                ["α",        f"{new_alpha:.8f}"],
            ]
            print(tabulate(table, headers=["Metric", "Value"], tablefmt="github"))
        
        

    return model, scaler, results

# ─────────────────────  DB 저장  ─────────────────────

def save_results_bulk(results: List[BatchResult]):
    objs = []
    for r in results:
        if any(np.isnan([r.mape, r.mae, r.rmse, r.r2, r.reward])):
            LOGGER.warning("Skip save: NaN found in metrics")
            continue
        objs.append(
            AI_REWARD(
                MAPE   = min(r.mape,   9999.999999),
                MAE    = min(r.mae,    9999.999999),
                RMSE   = min(r.rmse,   9999.999999),
                R2     = max(min(r.r2, 9999.999999), -9999.999999),
                REWARD = min(r.reward, 9999.999999),
            )
        )
    if objs:
        AI_REWARD.objects.bulk_create(objs)
        LOGGER.info(f"✅ Saved {len(objs)} rows to AI_REWARD")
from pathlib import Path
from pathlib import Path
import pandas as pd

MODEL_PATH  = Path("models/price_model.pkl")
SCALER_PATH = Path("models/price_scaler.pkl")

# ─────────────  debug switch  ──────────────
DEBUG_FIRST_BATCH = False         # ← 필요할 때만 True

# ─────────────────────  메인  ─────────────────────
def main():
    ticker = "QQQ"
    while True:
        # 1) 최신 배치
        batches = list(walk_forward_batches(mini_batch_normalization(ticker)))

        # 🔍 첫 배치 디버그 ----------------------------------
        if DEBUG_FIRST_BATCH and batches:
            train_b, pred_b = batches[0]

            print("\n🟡  [디버그] 첫 batch-train head()")
            print(train_b.head())
            print("\n🟡  [디버그] 첫 batch-pred head()")
            print(pred_b.head())

            sma_cols = [c for c in train_b.columns if c.startswith("SMA_")]
            print("\n🟢  포함된 SMA 컬럼:", sma_cols, "\n")
            input("▶ 엔터를 누르면 학습을 계속합니다…")

        # 2) 모델·스케일러 로드 (없으면 새로)
        model  = load_model(MODEL_PATH) if MODEL_PATH.exists() else SGDRegressor()
        scaler = load_or_init_scaler(SCALER_PATH, FIT_COLS)

        # 3) 학습
        model, scaler, results = partial_fit_batches(
            batches, model=model, scaler=scaler, verbose=True
        )

        # 4) 저장
        save_model(model, MODEL_PATH)
        save_scaler(scaler, SCALER_PATH)
        save_results_bulk(results)

        print("\n✅  모델·스케일러 저장 완료\n")



# ─────────────────────  엔트리  ─────────────────────
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted – shutting down cleanly")
