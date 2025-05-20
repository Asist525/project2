"""
Bi-LSTM + Attention + Monte-Carlo (CPU-only, dynamic LR schedule)
파일명: lstm_mc_cpu_dynamic.py
"""
from __future__ import annotations
import os, json, logging, sys
import numpy as np
import pandas as pd

# ─── 호환성 패치 ───────────────────────────────────────────
if not hasattr(np, "NaN"):
    np.NaN = np.nan
# ──────────────────────────────────────────────────────────

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.losses import Huber
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv1D, LSTM, Dense, Dropout, BatchNormalization, 
    LayerNormalization, Input, Bidirectional, MultiHeadAttention, Add, Concatenate
)
import matplotlib.pyplot as plt
import matplotlib
import pandas_ta as ta   # 기술적 지표
from nomalization import *
matplotlib.use("Agg")    # GUI 없는 환경 대비
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
tf.config.optimizer.set_jit(False)




# ───────────────────── 1. 데이터 전처리 ─────────────────────
def add_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """단기 방향성 개선을 위한 TA 피처 추가"""
    df_ta = df.copy()

    # ── 기존 지표 ──────────────────────────────────
    df_ta.ta.rsi(length=14, append=True)
    df_ta.ta.sma(length=5, append=True)
    df_ta.ta.sma(length=20, append=True)

    # ── 초단기 가격 변화 (ROC·Momentum) ─────────────
    df_ta["1d_ROC"] = df_ta["Close"].pct_change(1)
    df_ta["3d_ROC"] = df_ta["Close"].pct_change(3)
    df_ta["5d_ROC"] = df_ta["Close"].pct_change(5)

    df_ta["1d_Momentum"] = df_ta["Close"].diff(1)
    df_ta["3d_Momentum"] = df_ta["Close"].diff(3)
    df_ta["5d_Momentum"] = df_ta["Close"].diff(5)

    # ── Williams %R(5) ────────────────────────────
    df_ta["Williams_R_5"] = ta.willr(
        df_ta["High"], df_ta["Low"], df_ta["Close"], length=5
    )

    # ── True Strength Index(TSI 5·20) ─────────────
    tsi = ta.tsi(df_ta["Close"], fast=5, slow=20)      # pandas-ta 객체
    # pandas-ta 버전에 따라 Series 또는 DataFrame 반환
    if isinstance(tsi, pd.DataFrame):
        df_ta["TSI"] = tsi.iloc[:, 0]                  # 첫 컬럼만 사용
    else:
        df_ta["TSI"] = tsi

    # ── Money Flow Index(5) ────────────────────────
    df_ta["MFI_5"] = ta.mfi(
        df_ta["High"], df_ta["Low"], df_ta["Close"], df_ta["Volume"], length=5
    )

    # ── Volume Delta ───────────────────────────────
    df_ta["1d_Vol_Delta"] = df_ta["Volume"].diff(1)
    df_ta["3d_Vol_Delta"] = df_ta["Volume"].diff(3)

    # ── NaN·∞ 정리 ────────────────────────────────
    df_ta.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_ta.dropna(inplace=True)

    return df_ta





def load_and_preprocess(
        ticker: str,
        lookback: int = 365,
        end_date: str | pd.Timestamp | None = None
) -> pd.DataFrame:
    """
    • lookback: 뒤에서부터 가져올 영업일 수
    • end_date: 해당 날짜(포함) 기준으로 lookback 만큼 잘라서 반환
                None 이면 가장 최신 일자를 end_date 로 간주
    """
    from nomalization import trades_to_dataframe  # 프로젝트 util

    # ── ① 원본 로드 ────────────────────────────────────────────
    df = trades_to_dataframe(ticker)
    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    # ── ② 범위 슬라이싱 (end_date + lookback) ────────────────
    if end_date is None:
        end_date = df.index.max()
    end_date = pd.Timestamp(end_date)

    start_idx = end_date - pd.tseries.offsets.BDay(lookback - 1)
    if end_date.tzinfo is None:
        end_date = end_date.tz_localize("UTC")

    start_idx = end_date - pd.tseries.offsets.BDay(lookback - 1)

    # start_idx 역시 tz-aware 로 맞추기
    start_idx = start_idx.tz_convert("UTC") if start_idx.tzinfo else start_idx.tz_localize("UTC")

    df = df.loc[start_idx:end_date]        # ← 이제 tz 일치

    # ── ③ 전처리 ─────────────────────────────────────────────
    df.drop(columns=["Ticker"], inplace=True, errors="ignore")
    df["LogRet"] = np.log(df["Close"].pct_change() + 1)

    df["Prev_Close"] = df["Close"].shift(1)
    df = df.assign(
        SMA_5=df["Close"].rolling(5).mean(),
        SMA_20=df["Close"].rolling(20).mean(),
    )

    df = add_ta_features(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    return df

def scale_features(df: pd.DataFrame, cols: list[str]):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[cols])
    return scaled.astype(np.float32), scaler


def dynamic_weighted_huber(y_true, y_pred, alpha=0.95, delta=0.5, kappa=1.3):
    """
    • 시계열 각 시점에 대해 점진적 가중치 적용
    • 방향성 오판 페널티 완화
    """
    # 가중치 벡터 (단기 ~ 장기)
    steps = tf.shape(y_true)[1]
    weights = tf.linspace(1.0, alpha, steps)
    weights = tf.reshape(weights, (1, -1))  # (1, steps)

    # Huber 손실 계산
    error = y_true - y_pred
    abs_err = tf.abs(error)
    huber_loss = tf.where(abs_err < delta, 0.5 * tf.square(error), delta * (abs_err - 0.5 * delta))

    # 방향성 페널티 (부호가 다른 경우 kappa 배 가중치)
    direction_penalty = tf.cast(tf.sign(y_true) != tf.sign(y_pred), tf.float32)
    amplified_loss = huber_loss * (1 + direction_penalty * (kappa - 1))

    # 시점별 가중치 적용
    weighted_loss = weights * amplified_loss

    # 최종 평균 손실 반환
    return tf.reduce_mean(weighted_loss)



def create_sequences(data: np.ndarray, target: np.ndarray, ts: int, pred_days: int):
    """
    시계열 데이터에서 (X, y) 시퀀스를 생성
    • ts: 입력 시퀀스 길이
    • pred_days: 출력 시퀀스 길이
    """
    X, y = [], []
    for i in range(len(data) - ts - pred_days + 1):
        X.append(data[i:i + ts])
        y.append(target[i + ts:i + ts + pred_days])

    # np.stack → Ragged array 방지
    return (np.stack(X).astype(np.float32),
            np.stack(y).astype(np.float32))


# ───────────────────── 2. 모델 ─────────────────────

def build_model(ts: int, n_feat: int, pred_days: int, total_steps: int):
    """
    모델 구조 (짧은, 중간, 긴 Lookback 통합)
    """
    inp = Input(shape=(ts, n_feat), name="Input_Layer")

    # ── ① 짧은 Lookback (20) - 2층 ─────────────────
    short = inp[:, -20:, :]
    x_short = Conv1D(16, 3, activation="relu", padding="causal", name="Short_Conv")(short)
    x_short = BatchNormalization(name="Short_BN")(x_short)
    x_short = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2), name="Short_BiLSTM_1")(x_short)
    x_short = Bidirectional(LSTM(32, return_sequences=True, dropout=0.2), name="Short_BiLSTM_2")(x_short)
    x_short = LSTM(32, return_sequences=False, dropout=0.2, name="Short_LSTM")(x_short)

    # ── ② 중간 Lookback (60) - 2층 ─────────────────
    mid = inp[:, -60:, :]
    x_mid = Conv1D(32, 5, activation="relu", padding="causal", name="Mid_Conv")(mid)
    x_mid = BatchNormalization(name="Mid_BN")(x_mid)
    x_mid = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2), name="Mid_BiLSTM_1")(x_mid)
    x_mid = Bidirectional(LSTM(64, return_sequences=True, dropout=0.2), name="Mid_BiLSTM_2")(x_mid)
    x_mid = LSTM(64, return_sequences=False, dropout=0.2, name="Mid_LSTM")(x_mid)

    # ── ③ 긴 Lookback (120) - 2층 ─────────────────
    long = inp[:, -120:, :]
    x_long = Conv1D(64, 7, activation="relu", padding="causal", name="Long_Conv")(long)
    x_long = BatchNormalization(name="Long_BN")(x_long)
    x_long = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2), name="Long_BiLSTM_1")(x_long)
    x_long = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2), name="Long_BiLSTM_2")(x_long)
    x_long = LSTM(128, return_sequences=False, dropout=0.2, name="Long_LSTM")(x_long)

    # ── ④ Feature Merge (2층) ──────────────────────
    merged = Concatenate(name="Merged_Features")([x_short, x_mid, x_long])
    x = Dense(256, activation="relu", name="Dense_1")(merged)
    x = Dropout(0.3, name="Dropout_1")(x)
    x = BatchNormalization(name="BN_1")(x)
    x = Dense(128, activation="relu", name="Dense_2")(x)
    x = Dropout(0.3, name="Dropout_2")(x)
    out = Dense(pred_days, name="Output_Layer")(x)

    # ── ⑤ Optimizer & Loss ─────────────────────────
    decay_steps = int(total_steps * 0.8)
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=decay_steps,
        alpha=1e-2
    )
    opt = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-4,
        clipnorm=1.0
    )
    
    model = tf.keras.Model(inp, out, name="Multi_Path_LSTM_Model")
    model.compile(
        optimizer=opt,
        loss=lambda y, ŷ: sign_aware_huber(y, ŷ, delta=0.5, kappa=1.4),  # kappa 수정
        metrics=["mae", "mape"]
    )
    
    return model




def sign_aware_huber(y_true, y_pred, delta=0.5, kappa=1.3):
    """
    Huber 손실 + 방향성 패널티 (완화)
    """
    # 기본 Huber 손실
    error = y_true - y_pred
    abs_error = tf.abs(error)
    huber_loss = tf.where(abs_error < delta, 
                          0.5 * tf.square(error), 
                          delta * (abs_error - 0.5 * delta))
    
    # 방향성 패널티 (부호가 다르면 kappa 배)
    direction_mismatch = tf.cast(tf.sign(y_true) != tf.sign(y_pred), tf.float32)
    adjusted_loss = huber_loss * (1 + direction_mismatch * (kappa - 1))

    # 배치 차원만 평균 (pred_days 축)
    return tf.reduce_mean(adjusted_loss, axis=-1)





# ───────────────────── 3. 메인 ─────────────────────
def main(): 
    ticker = "QQQ"
    lookback = 412
    ts_len = 120
    horizon = 30
    end_date = "2024-07-18"
    
    # 데이터 로딩 및 전처리
    df = load_and_preprocess(ticker, lookback, end_date)
    feat_cols = [c for c in df.columns if c not in ("Close", "LogRet", "Prev_Close")]
    Xs, scX = scale_features(df, feat_cols)
    ys, scY = scale_features(df, ["LogRet"])
    X, y = create_sequences(Xs, ys.flatten(), ts_len, horizon)

    # 교차 검증
    tscv = TimeSeriesSplit(5)
    folds = list(tscv.split(X))
    
    # 마지막 Fold는 검증용으로만 사용
    train_folds = folds[:-1]   # 앞 4개 (훈련)
    val_fold = folds[-1]       # 최신 1개 (검증)

    best_val_loss = np.inf
    best_model = None
    
    # ── ① 훈련 (최신 fold 제외) ─────────────
    for fold_num, (tr_idx, _) in enumerate(train_folds, 1):
        print(f"\n🌀 Fold {fold_num}/{len(train_folds)} - Training Start")
        model = build_model(ts_len, len(feat_cols), horizon, int(np.ceil(len(tr_idx) / 32)) * 150)

        # 10에폭 단위 손실 로그 출력
        best_train_loss = np.inf
        for epoch in range(1, 151):
            history = model.fit(
                X[tr_idx], y[tr_idx],
                epochs=1, batch_size=32, verbose=0
            )

            # 손실 출력
            tr_loss = history.history["loss"][-1]
            if epoch % 10 == 0:
                print(f"Fold {fold_num} | Epoch {epoch} - Training Loss: {tr_loss:.6f}")

            # 모델 업데이트 (검증 fold에서만)
            if tr_loss < best_train_loss:
                best_train_loss = tr_loss
                print(f"🟢 Training Model Updated (Best Training Loss: {best_train_loss:.6f})")
    
    # ── ② 최종 검증 (최신 fold) ─────────────
    print("\n🧪 Final Fold - Validation Start")
    tr, va = val_fold
    model = build_model(ts_len, len(feat_cols), horizon, int(np.ceil(len(tr) / 32)) * 150)

    for epoch in range(1, 151):
        history = model.fit(
            X[tr], y[tr], validation_data=(X[va], y[va]),
            epochs=1, batch_size=32, verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
        )
        
        # 손실 출력
        val_loss = history.history["val_loss"][-1]
        if epoch % 10 == 0:
            print(f"Final Fold | Epoch {epoch} - Validation Loss: {val_loss:.6f}")

        # 모델 업데이트 (검증 단계에서만)
        if val_loss < best_val_loss:
            best_val_loss, best_model = val_loss, model
            print(f"🔵 Model Updated (Best Validation Loss: {best_val_loss:.6f})")

    # 최적 모델로 예측
    last_window = Xs[-ts_len:].reshape(1, ts_len, -1)
    next_logrets = best_model.predict(last_window)[0]
    next_logrets_pred = scY.inverse_transform(next_logrets.reshape(-1, 1)).flatten()
    last_price = df["Close"].iloc[-1]
    next_prices = last_price * np.exp(next_logrets_pred.cumsum())

    # 시각화
    fc_dates = pd.date_range(end=df.index[-1], periods=horizon + 1, freq="B")[1:]
    plt.figure(figsize=(14, 8))
    plt.plot(fc_dates, next_prices, label="Forecast", lw=2)
    plt.title(f"{ticker} – 30-Day Forecast")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig("forecast.png", dpi=300)
    plt.close()

    print("Next Prices:", next_prices)

if __name__ == "__main__":
    main()


