"""
Created on Tue Feb 15 07:56:54 2022
"""
#kis_api module 을 찾을 수 없다는 에러가 나는 경우 sys.path에 kis_api.py 가 있는 폴더를 추가해준다.
import kis_auth as ka 
from kis_auth import changeTREnv, getTREnv, read_token
import kis_domstk as kb
import kis_ovrseastk as kc
import pandas as pd
import sys
import yfinance as yf
# 토큰 발급
ka.auth()

# 실전투자 = prod, 모의투자 = vps 
# changeTREnv(token_key=token_key, svr='vps', product='01')



#-------------------------- 한국 거래 --------------------------------------------------------------------------------------------------------

# order: buy/sell   stock: 종목번호 6자리   qty: 매수 양    unpr: 지정가격
def korea_order_trade(order, stock_id, amount, cash):
    rt_data = kb.get_order_cash(ord_dv=order,itm_no=stock_id, qty=amount, unpr=cash)
    return rt_data

# 매수 가능양
def korea_get_inquire_psbl_order():
    rt_data = kb.get_inquire_psbl_order(pdno="", ord_unpr=0)
    ord_psbl_cash_value = rt_data.loc[0, 'ord_psbl_cash'] # ord_psbl_cash	주문가능현금
    ord_psbl_cash_value = rt_data.loc[0, 'nrcvb_buy_amt'] # nrcvb_buy_amt	미수없는매수가능금액
    return rt_data.loc[0, 'ord_psbl_cash']

# 주식잔고조회 (잔고현황)
def korea_account_amount():
    rt_data = kb.get_inquire_balance_obj()
    return rt_data

# 주식잔고조회 (보유종목리스트)
def korea_account_cash():
    rt_data = kb.get_inquire_balance_lst()
    print(rt_data)

# 주식잔고조회_실현손익(현재)
def korea_check_profit():
    rt_data = kb.get_inquire_balance_rlz_pl_obj()
    print(rt_data)
    rt_data = kb.get_inquire_balance_rlz_pl_lst()
    print(rt_data)


# 거래별 과거내역
def korea_check_history_profit():
    rt_data = kb.get_inquire_period_trade_profit_obj()
    print(rt_data)
    rt_data = kb.get_inquire_period_trade_profit_lst()
    print(rt_data)

# 거래별 과거내역 합산
def korea_check_history_profit_sum():
    rt_data = kb.get_inquire_period_profit_obj()
    print(rt_data)
    rt_data = kb.get_inquire_period_profit_lst()
    print(rt_data)



#-------------------------- 미국 거래 --------------------------------------------------------------------------------------------------------
# 거래
def global_order_trade(order, stock_id, amount, cash):
    rt_data = kc.get_overseas_order(ord_dv="buy", excg_cd="NASD", itm_no="NVDA", qty=1, unpr=123.3)
    return rt_data

# 증거금 통화별 조회
def global_get_inquire_psbl_order():

    rt_data = kc.get_overseas_inquire_foreign_margin()
    return rt_data

# 해외주식 잔고 현황
def global_get_inquire_psbl_order():
    rt_data = kc.get_overseas_inquire_balance(excg_cd="NASD", crcy_cd="")
    return rt_data

# 해외주식 잔고 내역
def global_account_cash():
    rt_data = kc.get_overseas_inquire_balance(excg_cd="NASD", crcy_cd="")
    return rt_data

# 거래별 과거내역(개별)
def global_check_history_profit(start, end):
    rt_data = kc.get_overseas_inquire_period_trans(excg_cd="", dvsn="", itm_no="", st_dt=start, ed_dt=end)
    return rt_data

# 거래별 과거내역(합산)
def global_check_history_profit_sum(start, end):
    rt_data = kc.get_overseas_inquire_period_trans_output2(excg_cd="", dvsn="", itm_no="", st_dt=start, ed_dt=end)
    return rt_data







#-------------------------- 주가 확인 --------------------------------------------------------------------------------------------------------

# 종목 시고저종

def get_stock_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        
        if not data.empty:
            current_price = data["Close"][-1]
            open_price = data["Open"][-1]
            high_price = data["High"][-1]
            low_price = data["Low"][-1]
            volume = data["Volume"][-1]
            
            return {
                "ticker": ticker,
                "current_price": current_price,
                "open_price": open_price,
                "high_price": high_price,
                "low_price": low_price,
                "volume": volume
            }
        else:
            return {"error": f"No data available for ticker: {ticker}"}
    except Exception as e:
        return {"error": str(e)}


# 기간 종목 시고저종
def get_stock_price(ticker, period="1mo", interval="1d"):
    """
    Get stock price data for a specific period and interval.
    
    Parameters:
    - ticker (str): The stock ticker symbol (e.g., "AAPL", "005930.KS").
    - period (str): The duration of the data (e.g., "1d", "1mo", "1y", "5y", "max").
    - interval (str): The data interval (e.g., "1d", "1wk", "1mo").
    
    Returns:
    - DataFrame: A DataFrame containing the stock price data.
    """
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

# 삼성전자 (005930.KS) 주가 1년치 일봉 데이터




#-------------------------------------유틸----------------------------------------------
def holiday(day):
    rt_data = kb.get_quotations_ch_holiday(dt=day)
    print(rt_data)




#-------------------------------------메인함수----------------------------------------------


ticker = "005930"


df = get_stock_price(f"{ticker}.KS", "max", "1d")
print(f"{ticker}종목 현재가: {df["Close"].iloc[len(df)-1]}")

df2 = global_get_inquire_psbl_order()
print(f"현재 미국 투자금: {df2["frcr_pchs_amt1"].iloc[0]}")
print(f"현재 미국 총 금액 : {df2["tot_evlu_pfls_amt"].iloc[0]}")
print(f"미국 수익금: {df2["ovrs_tot_pfls"].iloc[0]}")

profit = float(df2['tot_pftrt'].iloc[0])
print(f"미국 수익률: {profit:.2f}%")








# 한국주식 예수금만큼 매수
# amount = korea_get_inquire_psbl_order()
# price = df["Close"].iloc[len(df)-1]
# korea_order_trade("buy", "005930", int(float(amount) / float(price)), df["Close"].iloc[len(df)-1])


