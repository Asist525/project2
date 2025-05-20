import pandas as pd
import sys
import yfinance as yf
import os
import django
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
django.setup()
from history.models import Trade
from django.db.models import Count
def get_stock_price(ticker, period="max", interval="1d"):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval=interval)
        
        if not data.empty:
            data.reset_index(inplace=True)
            
            return data
        else:
            print(f"No data available for ticker: {ticker}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

def save_data(ticker):
    
    #기존 데이터 삭제
    deleted_count, _ = Trade.objects.filter(Ticker=ticker).delete()
    # 데이터 가져오기
    df = get_stock_price(ticker)
    
    # 필요한 열 선택
    data = df.loc[:, ["Date", "Open", "High", "Low", "Close", "Volume"]]
    data.insert(1, "Ticker", ticker)
    
    # 데이터프레임을 Django 모델 인스턴스로 변환
    trades = [
        Trade(
            Date=row["Date"],
            Ticker=row["Ticker"],
            Open=row["Open"],
            High=row["High"],
            Low=row["Low"],
            Close=row["Close"],
            Volume=row["Volume"]
        )
        for _, row in data.iterrows()
    ]
    
    # DB 저장
    Trade.objects.bulk_create(trades)


def main():
    
    #종목 저장
    ticker_list = ["AAPL", "QQQ"]
    for ticker in ticker_list:
       save_data(ticker)


    
    # # 종목 서치
    # ticker = "AAPL"
    # apple_trades = Trade.objects.filter(Ticker=ticker)



    # for trade in apple_trades:
    #     print(trade)
        
main()