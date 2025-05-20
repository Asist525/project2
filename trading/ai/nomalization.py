from __future__ import annotations
import pandas as pd
from pandas.tseries.offsets import CustomBusinessDay
import sys
import os
import django
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
import numpy as np

# Django 프로젝트 설정
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()
from history.models import Trade
import random
import matplotlib.pyplot as plt
import joblib
from datetime import timedelta
import logging
logger = logging.getLogger(__name__)
import random
KR_US_BDAY = CustomBusinessDay(calendar="KRX")  # 필요 시 NYSE 일정 병합
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



## 전역 변수 / 향후 보조지표 추가시 변경 예정
fit_cols = ["Open","High","Low","Volume","Prev_Close"] + [f"SMA_{p}" for p in (5,20,60)]



# Django 모델에서 주가 데이터를 불러오는 함수
def trades_to_dataframe(ticker):
    trades = list(Trade.objects.filter(Ticker=ticker).values())
    if not trades:
        print(f"데이터가 존재하지 않습니다: {ticker}")
        return pd.DataFrame()
    df = pd.DataFrame(trades)

    # Date 형식 변환
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Float 변환 (Decimal to float)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    #print(f"{ticker} 데이터 로드 완료 (총 {len(df)} 개)")
    return df


# 데이터프레임을 개월 단위로 미니 배치로 분할하는 제너레이터 함수
# 디폴트: 4개월
def batch_generator(df, batch_month=4):
    if df.empty:
        print("데이터가 비어 있습니다.")
        return

    # 날짜 필드가 없으면 중단
    if 'Date' not in df.columns:
        print("데이터프레임에 'Date' 필드가 필요합니다.")
        return

    # 날짜 기준으로 정렬
    df = df.sort_values(by='Date').reset_index(drop=True)

    # 월 단위로 분할
    start_date = df['Date'].min()
    end_date = df['Date'].max()
    current_date = start_date

    while current_date < end_date:
        # 다음 배치의 끝 날짜 계산
        next_date = current_date + pd.DateOffset(months=batch_month)
        
        # 현재 배치 추출
        batch = df[(df['Date'] >= current_date) & (df['Date'] < next_date)]
        
        if not batch.empty:
            yield batch
        
        # 다음 배치로 이동
        current_date = next_date


# 제너레이터로 생성된 배치를 셔플하는 함수
def shuffle_batches(generator):

    # 모든 배치를 일단 리스트에 로드
    batches = list(generator)
    
    # 셔플
    random.shuffle(batches)
    
    # 셔플된 배치를 다시 제너레이터로 반환
    for batch in batches:
        yield batch


# Mini-batch 단위로 MinMax Scaling만 적용 (배치별 독립 스케일링)
def normalize_batch(batch, columns=None, save_scaler_flag=True, scaler_filename="models/price_scaler.pkl"):

    if columns is None:
        columns = ["Open", "High", "Low", "Close", "Volume"]

    if batch.empty:
        return batch, None, None, None  # 빈 배치면 그대로 반환

    # 스케일러 생성
    scaler = MinMaxScaler()

    # 배치별 최소, 최대값 저장 (역스케일링을 위해 필요)
    min_vals = batch.loc[:, columns].min()
    max_vals = batch.loc[:, columns].max()

    # 배치별 독립 스케일링
    scaled_values = scaler.fit_transform(batch.loc[:, columns])

    # 스케일링된 데이터 적용
    scaled_batch = batch.copy()
    scaled_batch.loc[:, columns] = scaled_values

    # ** 스케일러 저장 **
    if save_scaler_flag:
        if isinstance(scaler, (BaseEstimator, TransformerMixin)):
            save_scaler(scaler, scaler_filename)
        else:
            raise TypeError(f"저장하려는 객체가 스케일러가 아닙니다: {type(scaler)}")

    # 컬럼 이름 포함하여 반환
    return scaled_batch, scaler, min_vals, max_vals


# 학습된 스케일러를 디스크에 저장하는 함수
def save_scaler(scaler, filepath="models/price_scaler.pkl"):

    if not isinstance(scaler, (BaseEstimator, TransformerMixin)):
        raise TypeError("스케일러는 scikit-learn의 TransformerMixin을 상속해야 합니다.")
    
    # 폴더가 없으면 생성
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # 스케일러 저장
    joblib.dump(scaler, filepath)
    print(f"스칼라 저장 완료")


# 장된 스케일러를 로드하는 함수
def load_scaler(filepath="models/price_scaler.pkl"):

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"지정된 스케일러 파일이 없습니다: {filepath}")
    
    # 스케일러 로드
    scaler = joblib.load(filepath)

    # 스케일러 타입 확인
    if not isinstance(scaler, (BaseEstimator, TransformerMixin)):
        raise TypeError(f"로드된 객체는 스케일러가 아닙니다: {type(scaler)}")
    
    return scaler

# 모델 저장 함수
def save_model(model, filepath="models/price_model.pkl"):
    # BaseEstimator 상속 여부 체크
    if not isinstance(model, BaseEstimator):
        print(f"모델 타입이 잘못되었습니다: {type(model)}")
        raise TypeError("모델은 scikit-learn의 BaseEstimator를 상속해야 합니다.")

    # 모델 저장
    with open(filepath, "wb") as f:
        joblib.dump(model, filepath)
    print(f"모델 저장 완료")


# 저장된 모델을 로드하는 함수
def load_model(filepath="models/price_model.pkl"):

    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"지정된 모델 파일이 없습니다: {filepath}")
    
    # 모델 로드
    model = joblib.load(filepath)
    print(f"모델 로드 완료")

    # 모델 타입 확인
    if not isinstance(model, BaseEstimator):
        raise TypeError(f"로드된 객체는 모델이 아닙니다: {type(model)}")
    
    return model

def add_sma_features(df: pd.DataFrame, periods=(5, 20, 60)) -> pd.DataFrame:
    """
    단순이동평균(SMA) 컬럼을 df에 추가한다.
    예) periods=(5,20,60) → 'SMA_5', 'SMA_20', 'SMA_60' 컬럼 생성
    """
    df = df.sort_values("Date").copy()
    for p in periods:
        df[f"SMA_{p}"] = df["Close"].rolling(window=p, min_periods=1).mean()
    return df



def mini_batch_scaling(
    batch_generator,
    batch_month=3,
    visualize=True,
    scaler_filename="models/price_scaler.pkl",
    sma_periods=(5, 20, 60)      # ← 추가: SMA 기간
):
    # 0) 스케일링 대상 기본 컬럼 + SMA 컬럼 ---------------------
    base_cols = ["Open", "High", "Low", "Close", "Volume"]
    sma_cols  = [f"SMA_{p}" for p in sma_periods]
    columns   = base_cols + sma_cols         # 스케일 대상 최종 목록

    scaled_batches, batch_ranges = [], []

    for i, raw_batch in enumerate(batch_generator):
        if raw_batch.empty:
            continue

        # 1) SMA 컬럼 추가 -------------------------------------
        batch = add_sma_features(raw_batch, periods=sma_periods)

        # 2) 타입 변환 후 MinMax 스케일 ------------------------
        batch = batch.copy()
        batch.loc[:, columns] = batch.loc[:, columns].astype(float)

        save_scaler_flag = (i == 0)
        scaled_batch, scaler, min_vals, max_vals = normalize_batch(
            batch,
            columns=columns,
            save_scaler_flag=save_scaler_flag,
            scaler_filename=scaler_filename,
        )
        scaled_batches.append(scaled_batch)
        batch_ranges.append((min_vals, max_vals))

        if visualize and i % 10 == 0:
            print(f"\n[Batch {i+1}] {batch['Date'].min()} ~ {batch['Date'].max()}")
            print(scaled_batch[columns].head())

    return scaled_batches, batch_ranges



# 제너레이터 형식의 배치에서 (train_b, pred_b) 쌍을 생성.
def walk_forward_batches(batches, train_months=3, pred_months=1):
    # 제너레이터를 리스트로 변환
    batches = list(batches)
    # 셔플
    random.shuffle(batches)

    # 각 배치 내부 정렬
    for batch in batches:
        batch = batch.sort_values("Date").reset_index(drop=True)
        # 배치 쪼개기
        cursor = batch["Date"].min()
        end    = batch["Date"].max()


        from pandas.tseries.offsets import CustomBusinessDay

        KR_US_BDAY = CustomBusinessDay(calendar="KRX")
        
        while cursor < end:
            # 4개월 배치
            batch_end = cursor + pd.DateOffset(months=4)

            full_batch = batch[(batch["Date"] >= cursor) & (batch["Date"] < batch_end)]
            
            # 3개월 학습 + 1개월 예측
            train_end = cursor + pd.DateOffset(months=3)
            train_batch = full_batch[(full_batch["Date"] >= cursor) & (full_batch["Date"] < train_end)]

            pred_batch = full_batch[(full_batch["Date"] >= train_end) & (full_batch["Date"] < batch_end)]

            # 예측 데이터가 부족한 경우 생성
            if pred_batch.empty:
                pred_dates = pd.date_range(
                    start=train_end, 
                    end=batch_end - pd.DateOffset(days=1), 
                    freq=KR_US_BDAY
                )
                pred_batch = pd.DataFrame({
                    "Date": pred_dates,
                    "id": [None] * len(pred_dates),
                    "Ticker": [train_batch["Ticker"].iloc[0]] * len(pred_dates),
                    "Open": [None] * len(pred_dates),
                    "High": [None] * len(pred_dates),
                    "Low": [None] * len(pred_dates),
                    "Close": [None] * len(pred_dates),
                    "Volume": [None] * len(pred_dates),
                })

            # 빈 배치는 건너뛰기
            if train_batch.empty:
                break

            yield train_batch, pred_batch if not pred_batch.empty else None
            cursor = batch_end  # 4개월 단위 이동


# 그래프 그리는 함수 => 향후 Django로 전환 예정
def plot_predictions(y_pred, y_actual, dates=None, batch_index=1, filename="predictions.html"):
    """
    예측값과 실제값을 시각화하는 함수 (Plotly 버전, 웹 출력)
    """
    # 기본 x축: Sample Index
    x = np.arange(len(y_pred)) if dates is None else dates
    
    # 서브플롯 생성
    fig = make_subplots(rows=1, cols=1, subplot_titles=[f"Batch {batch_index} - Predictions vs Actuals"])

    # 예측값 플롯
    fig.add_trace(
        go.Scatter(x=x, y=y_pred, mode='lines+markers', name="Predicted", opacity=0.7),
        row=1, col=1
    )

    # 실제값 플롯 (있을 때만)
    if y_actual:
        fig.add_trace(
            go.Scatter(x=x, y=y_actual, mode='lines+markers', name="Actual", opacity=0.7),
            row=1, col=1
        )

    # 레이아웃 설정
    fig.update_layout(
        title_text=f"Batch {batch_index} - Predictions vs Actuals",
        xaxis_title="Date" if dates is not None else "Sample Index",
        yaxis_title="Price (USD)",
        legend_title="Legend",
        height=600,
        width=1200,
        xaxis_tickformat="%Y-%m-%d" if dates is not None else None  # 날짜 포맷
    )

    # 그래프 렌더링
    fig.show()

    print(f"✅ 차트가 생성되었습니다: {os.path.abspath(filename)}")


# 배치별로 독립적으로 스케일링된 배치를 역정규화 후 역스케일링하는 함수
def inverse_scale_batch(scaled_batch, min_vals, max_vals, columns=None, normalization_method=None, lambdas=None):

    if columns is None:
        columns = ["Open", "High", "Low", "Close", "Volume"]
    
    # 스케일링된 데이터만 추출
    scaled_values = scaled_batch.loc[:, columns].values
    
    # 역정규화 (필요한 경우)
    if normalization_method == "log":
        scaled_values = np.exp(scaled_values)
    elif normalization_method == "log1p":
        scaled_values = np.expm1(scaled_values)
    elif normalization_method == "boxcox":
        if lambdas is None or not isinstance(lambdas, dict):
            raise ValueError("Box-Cox 복원을 위해 각 피처의 람다 값이 필요합니다.")
        
        restored_values = np.zeros_like(scaled_values)
        for i, col in enumerate(columns):
            lambda_val = lambdas.get(col)
            if lambda_val is None:
                raise ValueError(f"Box-Cox 복원을 위해 '{col}'의 람다 값이 필요합니다.")
            if lambda_val == 0:
                restored_values[:, i] = np.exp(scaled_values[:, i])  # λ=0 일 때는 로그 변환과 동일
            else:
                restored_values[:, i] = np.power(scaled_values[:, i] * lambda_val + 1, 1 / lambda_val)
        scaled_values = restored_values
    
    elif normalization_method == "yeo-johnson":
        if lambdas is None or not isinstance(lambdas, dict):
            raise ValueError("Yeo-Johnson 복원을 위해 각 피처의 람다 값이 필요합니다.")
        
        restored_values = np.zeros_like(scaled_values)
        for i, col in enumerate(columns):
            lambda_val = lambdas.get(col)
            if lambda_val is None:
                raise ValueError(f"Yeo-Johnson 복원을 위해 '{col}'의 람다 값이 필요합니다.")
            # Yeo-Johnson 복원
            if lambda_val == 0:
                restored_values[:, i] = np.exp(scaled_values[:, i]) - 1
            else:
                pos_idx = scaled_values[:, i] >= 0
                neg_idx = ~pos_idx
                restored_values[pos_idx, i] = np.power(scaled_values[pos_idx, i] * lambda_val + 1, 1 / lambda_val) - 1
                restored_values[neg_idx, i] = 1 - np.power(-(scaled_values[neg_idx, i]) * lambda_val + 1, 1 / lambda_val)
        scaled_values = restored_values
    
    # 배치별 최소-최대값을 사용한 역스케일링
    original_values = scaled_values * (max_vals.values - min_vals.values) + min_vals.values
    
    # 복원된 데이터프레임 생성 (날짜 유지)
    restored_batch = scaled_batch.copy()
    restored_batch.loc[:, columns] = original_values
    
    return restored_batch


# 복원된 예측값과 실제값을 비교하는 함수
def compare_predictions_with_actuals(restored_batch, original_df, columns=None):

    if columns is None:
        columns = ["Open", "High", "Low", "Close", "Volume"]
    
    # 날짜와 티커를 기준으로 병합
    merged_df = pd.merge(
        restored_batch,
        original_df[["Date", "Ticker"] + columns],
        on=["Date", "Ticker"],
        suffixes=("_pred", "_actual")
    )

    # 비교 출력
    print("\n예측값 vs 실제값 (일부):")
    print(merged_df.head())
    
    return merged_df


# 실제 예측
def predict_real_data(
    ticker: str,
    model: BaseEstimator,
    *,
    days_ahead: int = 21,
    scaler: MinMaxScaler | None = None,
    scaler_filename: str = "models/price_scaler.pkl",
    lookback_months: int = 3,
    calendar: CustomBusinessDay = KR_US_BDAY,
) -> pd.DataFrame:
    """최근 lookback_months 데이터로 스케일을 맞춘 뒤 days_ahead 영업일을 다단 예측."""

    # 0) 스케일러 확보 ─────────────────────────────────────────
    scaler = scaler or load_or_init_scaler(scaler_filename, fit_cols)

    # 1) 최근 n개월 데이터 로드 ────────────────────────────────
    df = trades_to_dataframe(ticker)
    if df.empty:
        logger.warning("(%s) 데이터 없음", ticker)
        return pd.DataFrame(columns=["Date", "Pred_Close"])

    cutoff = pd.Timestamp.utcnow() - pd.DateOffset(months=lookback_months)
    recent = df[df["Date"] >= cutoff].copy()
    if recent.empty:
        logger.warning("❌ (%s) 최근 %d개월 데이터 없음", ticker, lookback_months)
        return pd.DataFrame(columns=["Date", "Pred_Close"])

    # 2) Prev_Close 생성 ──────────────────────────────────────
    recent["Prev_Close"] = recent["Close"].shift(1).ffill()

    # 3) 스케일 & 피처 정렬 확인 ───────────────────────────────
    feature_cols = model.feature_names_in_.tolist()
    if feature_cols != fit_cols:
        raise ValueError(f"모델 피처 {feature_cols} ↔ 예상 {fit_cols} 불일치")

    # 4) 스케일러 fit 여부 확인 → 미-fit 시 최근 데이터로 fit ──
    if not hasattr(scaler, "data_min_"):
        scaler.fit(recent[feature_cols])
        joblib.dump(scaler, scaler_filename)
        print("🆕  스케일러를 최근 데이터로 fit & 저장")

    # 5) 시작행 준비 ───────────────────────────────────────────
    X_scaled = scaler.transform(recent[feature_cols])
    last_row = X_scaled[-1].reshape(1, -1)          # (1, n_features)
    last_date = recent["Date"].max()
    idx_prev  = feature_cols.index("Prev_Close")
    n_noise   = idx_prev                           # Open~Volume 컬럼 개수

    preds, dates = [], []

    # 6) 다중 스텝 예측 루프 ───────────────────────────────────
    for _ in range(days_ahead):
        # 6-1) 예측 (DataFrame으로 전달해 경고 제거)
        last_row_df = pd.DataFrame(last_row, columns=feature_cols)
        next_scaled_close = float(model.predict(last_row_df)[0])

        # 6-2) full_scaled_row 작성 (Prev_Close 자리만 교체)
        full_scaled_row = last_row.copy()
        full_scaled_row[0, idx_prev] = next_scaled_close

        # 6-3) 역-스케일 → 실제 Close 값
        next_close = float(
            scaler.inverse_transform(full_scaled_row)[0, idx_prev]
        )
        preds.append(next_close)

        # 6-4) 다음 영업일 날짜
        last_date += calendar
        dates.append(last_date)

        # 6-5) 입력 행 업데이트
        last_row = full_scaled_row  # ← Prev_Close 포함 전체 피처 교체
        # 탐색 노이즈: Open·High·Low·Volume 에 작은 변동 추가
        noise = np.random.normal(0.0, 0.01, size=n_noise)
        last_row[0, :n_noise] += noise

    # 7) 결과 & 시각화 ─────────────────────────────────────────
    result = pd.DataFrame({"Date": dates, "Pred_Close": preds})
    logger.info("(%s) +%d 영업일 예측 완료", ticker, days_ahead)

    plot_predictions(
        y_pred=result["Pred_Close"].values,
        y_actual=[],            # 실제값 없음
        dates=result["Date"].tolist(),
        batch_index=1,
    )
    return result


def load_or_init_scaler(path: str, fit_cols: list[str]) -> MinMaxScaler:
    """스케일러를 로드하거나, 없으면 ‘나중에’ fit 할 빈 스케일러를 반환."""
    try:
        sc = load_scaler(path)
        if (not hasattr(sc, "feature_names_in_") or
            list(sc.feature_names_in_) != fit_cols):
            print("⚠️  스케일러-피처 불일치 → 새 스케일러 생성")
            sc = MinMaxScaler()
    except FileNotFoundError:
        print("🆕  스케일러 파일 없음 → 새 스케일러 생성")
        sc = MinMaxScaler()

    sc.feature_names_in_ = np.array(fit_cols)  # 이름만 미리 세팅
    return sc



# 셔플까지 마친 배치를 내보내는 모듈
def mini_batch_normalization(ticker):
    # ticker = "QQQ"

    # 1단계: 전체 데이터 프레임 만들기
    df = trades_to_dataframe(ticker)

    # 2단계: 배치 제너레이터
    batch_gen = batch_generator(df, batch_month=4)

    # 3단계: 배치 스케일링 + 정규화 (스케일링이 먼저)
    scaled_batches, batch_ranges = mini_batch_scaling(
        batch_gen,
        visualize=False               # save_scaler_flag 인자 ❌ 제거
    )
    # 4단계: 배치 순서 섞기
    shuffled_batches = list(shuffle_batches(iter(scaled_batches)))

    return shuffled_batches
    