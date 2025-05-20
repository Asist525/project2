# -*- coding: utf-8 -*-
"""
Batch 학습 + 예측 + 모델/스케일러 저장 (Refactored)
- 데이터 누수 방지 (Prev_Close)
- 온라인 러닝 안전성 강화 (alpha 갱신 방식 수정)
- 방향성(MCC) + 변화율(SMAPE) 복합 보상(0.7 : 0.3)
- pred 배치는 오직 평가에만 사용 (학습 X)
- bulk DB 저장
"""

from __future__ import annotations

# ─────────────────────  표준 라이브러리  ─────────────────────
import os, sys, logging
from dataclasses import dataclass
from typing import Iterable, List, Tuple
from pathlib import Path

# ─────────────────────  외부 라이브러리  ─────────────────────
import numpy as np
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import matthews_corrcoef
from tabulate import tabulate

# ─────────────────────  Django ORM bootstrap  ─────────────────────
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_DIR)
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()
from ai.models import AI_REWARD2

# ─────────────────────  내부 모듈  ─────────────────────
from nomalization import (
    walk_forward_batches,
    mini_batch_normalization,
    load_model, save_model,
    load_scaler, save_scaler,
    load_or_init_scaler,
    add_sma_features
)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)
pd.set_option("display.max_colwidth", None)

# ─────────────────────  상수·전역  ─────────────────────
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

np.random.seed(42)  # 재현성 확보 (운영 시 삭제)

EPS                     = 1e-8
ALPHA_MIN               = 1e-8
ALPHA_RESET             = 1e-4
DECAY_K                 = 0.5

# 보상 가중치: 0.7(MCC) · 0.3(SMAPE)
W_MCC, W_SMAPE          = 0.7, 0.3

EXTRA_SMA   = [f"SMA_{p}" for p in (5, 20, 60)]
FIT_COLS    = ["Open", "High", "Low", "Volume", "Prev_Close"] + EXTRA_SMA
TARGET_COL  = "Close"
KR_US_BDAY  = CustomBusinessDay(calendar="KRX")

# ─────────────────────  결과 구조체  ─────────────────────
@dataclass
class BatchResult:
    train_period: Tuple[str, str]
    pred_period: Tuple[str, str]
    mcc: float
    smape: float
    reward: float
    y_true: np.ndarray
    y_pred: np.ndarray

# ─────────────────────  보상·α 헬퍼  ─────────────────────

def _composite_reward(mcc: float, smape: float) -> float:
    """MCC(−1~1)·SMAPE(0~1) → 0~1 스케일 보상"""
    mcc_norm   = max(mcc, -1.0)  # 안정성
    smape_norm = max(min(smape, 1.0), 0.0)
    # SMAPE는 작을수록 좋기 때문에 (1 − smape) 로 뒤집음
    return W_MCC * ((mcc_norm + 1) / 2) + W_SMAPE * (1 - smape_norm)


def _update_alpha(cur_alpha: float, reward: float) -> float:
    """exp 감쇠로 규제 강도 업데이트, ALPHA_MIN 이하로는 잘라냄"""
    new_alpha = cur_alpha * np.exp(-DECAY_K * reward)
    return max(new_alpha, ALPHA_MIN)

# ─────────────────────  핵심 함수  ─────────────────────

def partial_fit_batches(
    batches: Iterable[Tuple[pd.DataFrame, pd.DataFrame]] ,
    model:  SGDRegressor | None = None,
    scaler: MinMaxScaler  | None = None,
    verbose: bool = True,
) -> Tuple[SGDRegressor, MinMaxScaler, list[BatchResult]]:

    # ── 1. 모델·스케일러 초기화 ─────────────────────────
    if model is None:
        model = SGDRegressor(
            loss="huber",
            epsilon=1.35,
            penalty="elasticnet",
            alpha=1e-6,          # 규제 강도
            l1_ratio=0.1,        # L1 10 %  /  L2 90 %
            learning_rate="invscaling",
            eta0=0.01,
            power_t=0.25,
            random_state=42,
        )

    # fit 1 × scaler → transform train / pred 에 동일 적용
    if scaler is None:
        scaler = MinMaxScaler()

    results: list[BatchResult] = []

    # ── 2. 배치 루프 ───────────────────────────────────
    for train_b, pred_b in batches:
        if train_b.empty or pred_b.empty:
            continue

        # (1) 피처 확장(SMA)  ──────────────────────────
        if "SMA_5" not in train_b.columns:
            train_b = add_sma_features(train_b, periods=(5, 20, 60))
            pred_b  = add_sma_features(pred_b,  periods=(5, 20, 60))

        # (2) 누수 차단용 Prev_Close  ────────────────
        train_b = train_b.copy()
        train_b["Prev_Close"] = train_b[TARGET_COL].shift(1)
        train_b.dropna(subset=["Prev_Close"], inplace=True)

        pred_b  = pred_b.copy()
        pred_b["Prev_Close"] = pred_b[TARGET_COL].shift(1)
        pred_b.dropna(subset=["Prev_Close"], inplace=True)

        # (3) 스케일러 점진 학습 + 변환 ───────────────
        scaler.partial_fit(train_b[FIT_COLS])

        X_train = pd.DataFrame(
            scaler.transform(train_b[FIT_COLS]),
            columns=FIT_COLS, index=train_b.index
        )
        y_train = train_b[TARGET_COL].values

        # 중요 피처 가중치(선택)  ----------------------
        X_train.loc[:, "Volume"] *= 3
        X_train.loc[:, ["SMA_5", "SMA_20", "SMA_60"]] *= 2

        # (4) 온라인 학습  -----------------------------
        model.partial_fit(X_train, y_train)

        # ──────────────────────────────────────────────
        # ■ 평가 단계 : pred_b 그대로 사용  ■
        # ──────────────────────────────────────────────
        if pred_b.empty:
            continue

        X_pred = pd.DataFrame(
            scaler.transform(pred_b[FIT_COLS]),
            columns=FIT_COLS, index=pred_b.index
        )
        X_pred.loc[:, "Volume"] *= 3
        X_pred.loc[:, ["SMA_5", "SMA_20", "SMA_60"]] *= 2

        y_true = pred_b[TARGET_COL].astype(float).values
        y_pred = model.predict(X_pred)

        # 유효성 필터
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if mask.sum() < 2:
            LOGGER.warning("❌ Too few valid samples → Batch skipped")
            continue
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        # (5) 평가 지표  ------------------------------
        with np.errstate(divide="ignore", invalid="ignore"):
            r_true = np.diff(np.log(np.maximum(y_true, EPS)))
            r_pred = np.diff(np.log(np.maximum(y_pred, EPS)))

        mcc   = matthews_corrcoef(np.sign(r_true), np.sign(r_pred))
        smape = np.mean(np.abs(r_pred - r_true) /
                        (np.abs(r_pred) + np.abs(r_true) + EPS))

        reward    = _composite_reward(mcc, smape)
        new_alpha = _update_alpha(model.alpha, reward)
        model.set_params(alpha=new_alpha)

        # (6) 결과 기록  ------------------------------
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
                mcc=mcc,
                smape=smape,
                reward=reward,
                y_true=y_true,
                y_pred=y_pred,
            )
        )

        if verbose:
            print(
                tabulate(
                    [
                        ["MCC",   f"{mcc:+.4f}"],
                        ["SMAPE", f"{smape*100:.2f}%"],
                        ["Reward",f"{reward:.4f}"],
                        ["α",     f"{new_alpha:.8f}"],
                    ],
                    headers=["Metric", "Value"],
                    tablefmt="github",
                )
            )

    return model, scaler, results

# ─────────────────────  DB 저장  ─────────────────────

def save_results_bulk(results: List[BatchResult]):
    objs = []
    for r in results:
        if any(np.isnan([r.mcc, r.smape, r.reward])):
            continue
        objs.append(
            AI_REWARD2(
                MCC    = max(min(r.mcc,  1.0), -1.0),
                SMAPE  = min(r.smape, 1.0),
                REWARD = r.reward,
            )
        )
    if objs:
        AI_REWARD2.objects.bulk_create(objs)
        LOGGER.info(f"✅ Saved {len(objs)} rows → AI_REWARD")

# ─────────────────────  메인  ─────────────────────
MODEL_PATH  = Path("models/price_model.pkl")
SCALER_PATH = Path("models/price_scaler.pkl")

def main():
    ticker = "QQQ"
    batches = list(walk_forward_batches(mini_batch_normalization(ticker)))
    DEBUG_FIRST_BATCH = False  # 🔄 디버그용 첫 배치 출력
    DEBUG_NOISE = True      # 🔄 노이즈 주입 디버그 (True 시 노이즈 제거)

    # 🔄 모델·스케일러 로드 (최초 1회)
    model  = load_model(MODEL_PATH) if MODEL_PATH.exists() else None
    scaler = load_or_init_scaler(SCALER_PATH, FIT_COLS)

    # 🔄 모델 초기화
    if model is None:
        print("\n⚠️  모델 초기화 중...")
        model = SGDRegressor(
            loss="huber",
            epsilon=1.35,
            random_state=42,
            penalty="l2",
            alpha=1e-4,
            learning_rate="invscaling",
            eta0=0.01,
            power_t=0.25,
            fit_intercept=True,
        )

    # 🔄 모델 초기 상태 디버그
    print("\n🔄 모델 초기 상태:")
    print(f"Alpha: {model.alpha}")
    print(f"Coef: {model.coef_[:5] if hasattr(model, 'coef_') else 'Not initialized'}")
    print(f"Intercept: {model.intercept_ if hasattr(model, 'intercept_') else 'Not initialized'}")

    # 🔄 배치 학습 시작
    while True:
        model, scaler, results = partial_fit_batches(batches, model=model, scaler=scaler, verbose=True)

        # 🔄 결과 디버그
        if results:
            sample_result = results[-1]
            print("\n📝  [디버그] 최근 배치 결과")
            print(f"Train Period: {sample_result.train_period}")
            print(f"Pred Period: {sample_result.pred_period}")
            print(f"MCC: {sample_result.mcc:.4f}")
            print(f"SMAPE: {sample_result.smape:.4f}")
            print(f"Reward: {sample_result.reward:.4f}")
            print(f"y_true[-5:]: {sample_result.y_true}")
            print(f"y_pred[-5:]: {sample_result.y_pred}")
    

        # 🔄 모델·스케일러 저장 (주기적 저장 고려)
        save_model(model, MODEL_PATH)
        save_scaler(scaler, SCALER_PATH)
        save_results_bulk(results)

        print("\n✅ 모델·스케일러 저장 완료\n")

        # 🔄 무한 루프 종료 옵션 (필요 시)
        if not DEBUG_NOISE:
            break


        

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted – shutting down cleanly")
