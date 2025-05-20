"""refactored_trading.py

자동 골든크로스 매매 스크립트 (한국투자증권 해외주식 API)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

import pandas as pd
import yfinance as yf
import pandas_ta as ta

import kis_auth as ka
from kis_auth import changeTREnv, getTREnv, read_token
import kis_ovrseastk as kc

pd.options.display.float_format = "{:.2f}".format
logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)s │ %(message)s")

# ────────────────────────────────────────────────────────── 기본 설정 ────
TICKER_LIST: List[str] = [
    "NVDA", "SMCI", "PLTR", "AMD", 
    "COIN", "SNOW", "SHOP", "TSLA"
]

@dataclass
class TraderConfig:
    svr: str = "prod"  # 실전: "prod", 모의: "vps"
    product_code: str = "01"

    token_key: str = field(init=False)

    def __post_init__(self) -> None:
        self.token_key = read_token()
        changeTREnv(token_key=self.token_key, svr=self.svr, product=self.product_code)
        ka.auth()
        logging.info("환경 설정 완료 · svr=%s", self.svr)

# ────────────────────────────────────────────────────── 유틸리티 함수 ────

def fetch_price(ticker: str, period: str = "3mo", interval: str = "1d") -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    data = stock.history(period=period, interval=interval)
    if data.empty:
        raise ValueError(f"{ticker} 가격 데이터를 찾을 수 없습니다.")
    data.reset_index(inplace=True)
    return data

def add_ta(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.ta.rsi(length=14, append=True)
    out["SMA_10"] = out["Close"].rolling(10).mean()
    out["SMA_50"] = out["Close"].rolling(50).mean()
    return out

# ─────────────────────────────────────────────────────────── Trader ────

class PortfolioTrader:
    def __init__(self, cfg: TraderConfig):
        self.cfg = cfg
        self.buy_list: List[str] = self._load_current_holdings()
        self.error_list: List[str] = []
        self.cash_per_trade: float = 400.0  # 종목당 고정 매수 금액
        logging.info("1회당 주문 가능 금액(고정): %.2f", self.cash_per_trade)

    def _load_current_holdings(self) -> List[str]:
        bal = kc.get_overseas_inquire_balance_lst()
        tickers = bal["ovrs_pdno"].tolist()
        logging.info("현재 보유 종목: %s", tickers)
        return tickers

    def _buy_signal(self, df: pd.DataFrame) -> bool:
        prev, curr = df.iloc[-2], df.iloc[-1]
        return prev["SMA_10"] < prev["SMA_50"] and curr["SMA_10"] > curr["SMA_50"]

    def _sell_signal(self, df: pd.DataFrame) -> bool:
        prev, curr = df.iloc[-2], df.iloc[-1]
        return prev["SMA_10"] > prev["SMA_50"] and curr["SMA_10"] < curr["SMA_50"]

    def run(self, tickers: List[str] = TICKER_LIST) -> None:
        for ticker in tickers:
            try:
                df = add_ta(fetch_price(ticker))
            except Exception as exc:
                logging.warning("%s 데이터 오류 → %s", ticker, exc)
                continue

            if len(df) < 60:
                logging.warning("데이터 부족: %s", ticker)
                continue

            price = round(df["Close"].iloc[-1], 2)
            qty = int(self.cash_per_trade // price)
            if qty == 0:
                logging.info("%s: 주문 수량 0 → 건너뜀", ticker)
                continue

            if self._buy_signal(df) and ticker not in self.buy_list:
                self._buy(ticker, qty, price)

            elif self._sell_signal(df) and ticker in self.buy_list:
                self._sell(ticker, qty, price)

            else:
                logging.debug("%s: 조건 불충족", ticker)

        logging.info("최종 보유 리스트: %s", self.buy_list)
        logging.info("오류 리스트: %s", self.error_list)

    def _buy(self, ticker: str, qty: int, price: float) -> None:
        logging.info("매수 신호 ▶ %s @ %.2f × %d", ticker, price, qty)
        resp = kc.get_overseas_order(
            ord_dv="buy", excg_cd="NASD", itm_no=ticker, qty=qty, unpr=price
        )
        if resp is None:
            logging.error("매수 실패: %s", ticker)
            self.error_list.append(ticker)
        else:
            self.buy_list.append(ticker)
            logging.info("매수 성공: %s", ticker)

    def _sell(self, ticker: str, qty: int, price: float) -> None:
        logging.info("매도 신호 ▶ %s @ %.2f × %d", ticker, price, qty)
        resp = kc.get_overseas_order(
            ord_dv="sell", excg_cd="NASD", itm_no=ticker, qty=qty, unpr=price
        )
        if resp is None:
            logging.error("매도 실패: %s", ticker)
            self.error_list.append(ticker)
        else:
            self.buy_list.remove(ticker)
            logging.info("매도 성공: %s", ticker)

# ─────────────────────────────────────────────────────────────── 실행 ────

def main() -> None:
    cfg = TraderConfig(svr="prod")  # 실계좌 운용 (주의)
    trader = PortfolioTrader(cfg)
    trader.run()

if __name__ == "__main__":
    main()
