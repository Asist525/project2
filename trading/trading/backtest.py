from __future__ import annotations

"""RSI / SMA-based back‑test with flexible buy & multi-strategy sell.

Adds support for both fixed-horizon (days) and indicator-based sell rules.
The original DataFrame (`df`) remains untouched; a separate `trades_df`
is produced containing per‑trade details (buy/sell dates, holding period,
gross & net returns). Prevents overlapping positions by enforcing a cooldown
period after each buy signal.
"""

from typing import List, Dict, Set
import numpy as np
if not hasattr(np, "NaN"):
    np.NaN = np.nan
import pandas as pd
import pandas_ta as ta
from tabulate import tabulate
import matplotlib.pyplot as plt
import yfinance as yf
pd.options.display.float_format = '{:.2f}'.format

# ────────────────────────── Global settings ────────────────────────── #
TICKER_LIST = [
    "NVDA",  # 반도체, AI
    "SMCI",  # 서버, 데이터센터
    "PLTR",  # 데이터 분석, AI
    "AMD",   # 반도체, 컴퓨팅
    "COIN",  # 암호화폐 거래소
    "SNOW",  # 클라우드 데이터 웨어하우스
    "SHOP",   # 전자상거래 플랫폼
    "TSLA"
]
# korea = EWY | U.S = QQQ | Russia = IWM | TSLA  | SMCI | PLTR | NVDA | IONQ | SONU
# > 600 => NVDA | TSLA

END_DATE = "2025-05-12"

RSI_LENGTH = 21
RSI_BUY_THRESHOLD = 30
DAYS_TO_CHECK: List[int] = [1, 5, 10, 20, 30, 60, 90, 180]

SELL_HOLD_DAYS = 10  # Fixed holding period (optional)

SELL_STRATEGY = "SMA_DEAD_CROSS_10_50"  
# 'FIXED' | 'RSI_OVERBOUGHT_70' | 'SMA_DEAD_CROSS_5_20' | 'SMA_DEAD_CROSS_10_50'

BUY_COMMISSION = 0.25 / 100
SELL_COMMISSION = 0.25278 / 100
SEC_FEE = 0.00278 / 100

BUY_STRATEGY = "SMA_GOLDEN_CROSS_10_50"  
# 'RSI_30_CROSS' | 'RSI_BELOW_30' | 'RSI_OVERBOUGHT_70' | 'SMA_GOLDEN_CROSS_5_20' | 'SMA_GOLDEN_CROSS_10_50'

# ────────────────────────── Helpers ────────────────────────── #

def add_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.ta.rsi(length=RSI_LENGTH, append=True)
    out["SMA_5"] = out["Close"].rolling(5).mean()
    out["SMA_10"] = out["Close"].rolling(10).mean()
    out["SMA_20"] = out["Close"].rolling(20).mean()
    out["SMA_50"] = out["Close"].rolling(50).mean()
    out["SMA_200"] = out["Close"].rolling(200).mean()
    return out

def fetch_price(ticker: str, period: str = "max", interval: str = "1d") -> pd.DataFrame:
    """Yahoo Finance로부터 가격 데이터 수집"""
    stock = yf.Ticker(ticker)

    data = stock.history(period=period, interval=interval)
    if data.empty:
        raise ValueError(f"{ticker} 가격 데이터를 찾을 수 없습니다.")
    data.reset_index(inplace=True)
    return data

def load_and_preprocess(ticker: str, *, end_date: str | pd.Timestamp | None = None) -> pd.DataFrame:
    df = fetch_price(ticker)
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df.set_index("Date", inplace=True)
    end_dt = pd.Timestamp(end_date or df.index.max(), tz="UTC")
    df = df.loc[:end_dt]
    df = add_ta_features(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


def get_buy_points(df: pd.DataFrame, thresh: float = RSI_BUY_THRESHOLD) -> pd.Index:
    rsi_col = f"RSI_{RSI_LENGTH}"
    if BUY_STRATEGY == "RSI_30_CROSS":
        return df[(df[rsi_col].shift(1) < thresh) & (df[rsi_col] >= thresh)].index
    if BUY_STRATEGY == "RSI_BELOW_30":
        return df[df[rsi_col] < thresh].index
    if BUY_STRATEGY == "RSI_OVERBOUGHT_70":
        return df[df[rsi_col] > 70].index
    if BUY_STRATEGY == "SMA_GOLDEN_CROSS_5_20":
        return df[(df["SMA_5"].shift(1) < df["SMA_20"].shift(1)) & (df["SMA_5"] >= df["SMA_20"])].index
    if BUY_STRATEGY == "SMA_GOLDEN_CROSS_10_50":
        return df[(df["SMA_10"].shift(1) < df["SMA_50"].shift(1)) & (df["SMA_10"] >= df["SMA_50"])].index
    if BUY_STRATEGY == "SMA_GOLDEN_CROSS_10_50_200":
        return df[
            (df["SMA_200"].shift(1) < df["SMA_50"].shift(1)) &  # 50일 SMA가 200일 SMA를 돌파
            (df["SMA_50"].shift(1) < df["SMA_10"].shift(1)) &  # 10일 SMA가 50일 SMA를 돌파
            (df["SMA_10"] >= df["SMA_50"])  # 현재 10일 SMA가 50일 SMA 이상
        ].index
    raise ValueError("Unknown BUY_STRATEGY")


def get_sell_point(df: pd.DataFrame, buy_idx: int) -> int:
    if SELL_STRATEGY == "FIXED":
        return buy_idx + SELL_HOLD_DAYS  # Fixed holding period
    elif SELL_STRATEGY == "RSI_OVERBOUGHT_70":
        for i in range(buy_idx + 1, len(df)):
            if df.iloc[i][f"RSI_{RSI_LENGTH}"] > 70:
                return i
    elif SELL_STRATEGY == "SMA_DEAD_CROSS_5_20":
        for i in range(buy_idx + 1, len(df)):
            if df.iloc[i]["SMA_5"] < df.iloc[i]["SMA_20"]:
                return i
    elif SELL_STRATEGY == "SMA_DEAD_CROSS_10_50":
        for i in range(buy_idx + 1, len(df)):
            if df.iloc[i]["SMA_10"] < df.iloc[i]["SMA_50"]:
                return i
    return -1  # No sell signal found

# ────────────────────────── Core analysis ────────────────────────── #

def compute_price_changes(df: pd.DataFrame, *, days: List[int] = DAYS_TO_CHECK) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    buy_pts = get_buy_points(df)
    horizon_records, trade_records, open_positions = [], [], set()

    for buy_dt in buy_pts:
        # Skip if an active position overlaps this buy signal
        if any((buy_dt >= pos) and (buy_dt < pos + pd.DateOffset(days=SELL_HOLD_DAYS)) for pos in open_positions):
            continue

        buy_px = df.loc[buy_dt, "Close"]
        buy_idx = df.index.get_loc(buy_dt)

        # Mark this position as active for the holding period
        open_positions.add(buy_dt)

        # horizon returns table
        row = {"Buy_Date": buy_dt}
        for d in days:
            tgt_idx = buy_idx + d
            row[f"{d}_days"] = np.nan if tgt_idx >= len(df) else round(((df.iloc[tgt_idx]["Close"] - buy_px) / buy_px * 100) - (BUY_COMMISSION+SELL_COMMISSION+SEC_FEE)*100, 2)
        horizon_records.append(row)

        # flexible sell strategy
        sell_idx = get_sell_point(df, buy_idx)
        if 0 <= sell_idx < len(df):
            sell_dt = df.index[sell_idx]
            sell_px = df.iloc[sell_idx]["Close"]
            gross = (sell_px - buy_px) / buy_px * 100
            net = gross - (BUY_COMMISSION+SELL_COMMISSION+SEC_FEE)*100
            trade_records.append({
                "Buy_Date": buy_dt,
                "Sell_Date": sell_dt,
                "Year": buy_dt.year,
                "Holding_Days": (sell_dt - buy_dt).days,
                "Gross_%": round(gross, 2),
                "Net_%": round(net, 2),
            })

    horizon_df = pd.DataFrame(horizon_records)
    trades_df = pd.DataFrame(trade_records)

    # 연도별 수익률 계산
    if not trades_df.empty:
        yearly_summary = trades_df.groupby("Year").agg(
            Total_Trades=("Net_%", "count"),
            Avg_Return=("Net_%", lambda x: round(x.mean(), 2)),
            Max_Return=("Net_%", lambda x: round(x.max(), 2)),
            Min_Return=("Net_%", lambda x: round(x.min(), 2)),
            Total_Return=("Net_%", lambda x: round(x.sum(), 2)),
            Win_Rate=("Net_%", lambda x: round((x > 0).mean() * 100, 2)),
            Std_Dev=("Net_%", lambda x: round(x.std(), 2)),
        ).reset_index()
    else:
        yearly_summary = pd.DataFrame()

    # 전체 평가지표 추가
    overall_metrics = pd.DataFrame({
        "Metric": ["Win Rate (%)", "Average (%)", "Max (%)", "Min (%)", "Total Return (%)", "Std Dev"],
        "Value": [
            round((trades_df["Net_%"] > 0).mean() * 100, 2),
            round(trades_df["Net_%"].mean(), 2),
            round(trades_df["Net_%"].max(), 2),
            round(trades_df["Net_%"].min(), 2),
            round(trades_df["Net_%"].sum(), 2),
            round(trades_df["Net_%"].std(), 2),
        ]
    })

    return horizon_df, trades_df, yearly_summary, overall_metrics

# ────────────────────────── Entry-point ────────────────────────── #

def run_backtest(start_price=100000, TICKER_LIST=TICKER_LIST):
    all_horizon_dfs = []
    all_trades_dfs = []
    all_yearly_summaries = []

    for TICKER in TICKER_LIST:
        # 데이터 로딩 및 전처리
        df = load_and_preprocess(TICKER, end_date=END_DATE)

        # 개별 종목의 백테스트
        horizon_df, trades_df, yearly_summary, overall_metrics = compute_price_changes(df)

        # 각 데이터프레임에 종목 이름 추가
        horizon_df["Ticker"] = TICKER
        trades_df["Ticker"] = TICKER
        yearly_summary["Ticker"] = TICKER

        # 리스트에 저장
        all_horizon_dfs.append(horizon_df)
        all_trades_dfs.append(trades_df)
        all_yearly_summaries.append(yearly_summary)


    # 전체 통합 데이터프레임 생성
    combined_yearly_summary = pd.concat(all_yearly_summaries, ignore_index=True)
    
    pivot_df = combined_yearly_summary.pivot_table(index="Year", columns="Ticker", values="Total_Return")
    pivot_df["Average"] = pivot_df.replace(0, np.nan).mean(axis=1)
    pivot_df.fillna(0, inplace=True)

    correlation_matrix = pivot_df.drop(columns=["Average"]).corr().round(3)
    scenarios = {
        "2008_Financial_Crisis": (2007, 2009),
        "2019_COVID_Pandemic": (2018, 2020),
        "2022_COVID_Long_Term": (2021, 2023),
        "2025_Trump_Reelection": (2024, 2025)
    }
    
    for scenario, (start_year, end_year) in scenarios.items():
        scenario_df = pivot_df[(pivot_df.index >= start_year) & (pivot_df.index <= end_year)]
        scenario_df.reset_index(inplace=True)
        print(f"\n🗓️ {scenario.replace('_', ' ')} Data")
        print(tabulate(scenario_df, headers="keys", tablefmt="fancy_grid", showindex=False, floatfmt=".4f"))

    
    
    volatilities = pivot_df.drop(columns=["Average"]).std()

    # 상관관계 행렬
    correlation_matrix = pivot_df.drop(columns=["Average"]).corr().round(3)

    # 포트폴리오의 변동성 계산
    weights = np.ones(len(volatilities)) / len(volatilities)  # 각 종목 동일 가중치 (예시)
    cov_matrix = correlation_matrix.values * np.outer(volatilities, volatilities)
    portfolio_volatility = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))

    # 개별 자산의 가중평균 변동성
    weighted_volatility = np.dot(weights, volatilities)

    # 분산 효과 계산
    diversification_benefit = (1 - (portfolio_volatility / weighted_volatility)) * 100

    # 결과 출력
    diversification_benefit = round(diversification_benefit, 2)

    summary_df = pd.DataFrame({
    "Metric": ["Mean", "Median", "Standard Deviation", "Variance", "Sum", "Min", "Max", "Range"],
    "Value": [
        pivot_df["Average"].mean(),
        pivot_df["Average"].median(),
        pivot_df["Average"].std(),
        pivot_df["Average"].var(),
        pivot_df["Average"].sum(),
        pivot_df["Average"].min(),
        pivot_df["Average"].max(),
        pivot_df["Average"].max() - pivot_df["Average"].min()
    ]
    })

    pivot_df_reset = pivot_df.reset_index()
    
    pivot_df_reset["Price"] = start_price * (1 + pivot_df_reset["Average"] / 100).cumprod()
    pivot_df_reset["Log_Average"] = np.log1p(pivot_df_reset["Price"])
    
    
    return summary_df, pivot_df, correlation_matrix, pivot_df_reset, diversification_benefit






def main():
    df1, df2, df3, df4, portfolio_volatility = run_backtest()
    print(df1)
    print(df2)
    print(df3)
    print(df4)
    print(portfolio_volatility)
    

if __name__ == "__main__":
    main()
