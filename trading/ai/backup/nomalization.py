import pandas as pd
import sys
import os
import django
from sklearn.preprocessing import RobustScaler, QuantileTransformer, StandardScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Django 프로젝트 설정
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()
from history.models import Trade


def trades_to_dataframe(ticker):
    """
    Django 모델에서 주가 데이터를 불러오는 함수
    """
    trades = list(Trade.objects.filter(Ticker=ticker).values())
    
    if not trades:
        print(f"❌ 데이터가 존재하지 않습니다: {ticker}")
        return pd.DataFrame()

    df = pd.DataFrame(trades)

    # Date 형식 변환
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Float 변환 (Decimal to float)
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = df[col].astype(float)

    print(f"✅ {ticker} 데이터 로드 완료 (총 {len(df)} 개)")
    return df

def add_sma_features(df: pd.DataFrame, periods=(5, 20, 60)) -> pd.DataFrame:
    """
    df 에 단순이동평균(SMA) 컬럼을 추가한다.
    periods 매개변수에 원하는 기간(영업일 수)을 tuple 로 넣어주면 됨.
    예: (5, 20, 60) → 'SMA_5', 'SMA_20', 'SMA_60' 컬럼 생성
    """
    df = df.sort_values("Date").copy()
    for p in periods:
        col = f"SMA_{p}"
        df[col] = df["Close"].rolling(window=p, min_periods=1).mean()
    return df

def mini_batch_scaling(
    df: pd.DataFrame,
    feature_cols=None,
    batch_size: int = 100,
    visualize: bool = False,
    sma_periods=(5, 20, 60),        # ← 추가: SMA 기간 설정
):
    """
    Mini-Batch StandardScaling + SMA 컬럼 추가
    """
    # 0) SMA 추가 -------------------------------------------------
    df = add_sma_features(df, periods=sma_periods)

    # 1) 스케일링 대상 컬럼 목록 정의 ----------------------------
    base_cols = ["Open", "High", "Low", "Close", "Volume"]
    sma_cols  = [f"SMA_{p}" for p in sma_periods]
    if feature_cols is None:
        feature_cols = base_cols + sma_cols

    # 2) 빈 데이터 체크 ------------------------------------------
    if df.empty or not set(feature_cols).issubset(df.columns):
        print("❌ 스케일 대상 컬럼이 비어 있거나 누락되었습니다.")
        return df, None

    # 3) Mini-Batch StandardScaler -------------------------------
    scaler = StandardScaler()
    scaled_array = np.zeros_like(df[feature_cols].values)

    for i in range(0, len(df), batch_size):
        batch = df[feature_cols].iloc[i:i + batch_size].values
        if i == 0:                           # 첫 배치에서 fit
            scaler.partial_fit(batch)
        scaled_array[i:i + batch_size] = scaler.transform(batch)

    # 4) 결과 DataFrame ------------------------------------------
    scaled_df = df.copy()
    scaled_df[feature_cols] = scaled_array

    # 5) 시각화 (옵션) -------------------------------------------
    if visualize:
        drawing_plotly(
            scaled_df[feature_cols],
            original_df=df[feature_cols],
            title="Mini-Batch StandardScaler + SMA"
        )

    return scaled_df, scaler


def drawing_plotly(scaled_df, original_df=None, title="Scaled Data"):
    """
    Plotly를 이용한 스케일링된 데이터 시각화 함수 (정규화 전후 비교)
    """
    columns = ["Open", "High", "Low", "Close", "Volume"]

    # 2열로 구성된 서브플롯 생성
    fig = make_subplots(rows=len(columns), cols=2, 
                        subplot_titles=[f"Before Scaling: {col}" for col in columns] + 
                                        [f"After Scaling: {col}" for col in columns])

    for i, col in enumerate(columns, 1):
        # 스케일링 전 (왼쪽)
        if original_df is not None:
            fig.add_trace(
                go.Histogram(x=original_df[col], nbinsx=100, opacity=0.6, marker_color="skyblue", name=f"Before Scaling: {col}"),
                row=i, col=1
            )

        # 스케일링 후 (오른쪽)
        fig.add_trace(
            go.Histogram(x=scaled_df[col], nbinsx=100, opacity=0.6, marker_color="steelblue", name=f"After Scaling: {col}"),
            row=i, col=2
        )

    # 레이아웃 설정
    fig.update_layout(height=3000, width=1500, title_text=title, showlegend=False)
    fig.show()





def main():
    ticker = "QQQ"
    data = trades_to_dataframe(ticker)

    # 데이터가 비어있으면 중단
    if data.empty:
        print("데이터가 비어있습니다. 스케일링을 중단합니다.")
        return
    
    # Mini-Batch Scaling
    scaled_df_mini_batch, mini_batch_scaler = mini_batch_scaling(data, batch_size=100, visualize=True)
    print(scaled_df_mini_batch.head())


if __name__ == "__main__":
    main()
