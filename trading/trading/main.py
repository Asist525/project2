from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from trading.backtest import run_backtest

app = FastAPI()


@app.get("/")
def health_check():
    """트레이딩 서비스 헬스체크"""
    return {"status": "ok", "message": "Trading service is running"}


@app.post("/run-backtest")
def run_backtest_endpoint():
    """
    백테스트 실행 API
    """
    try:
        summary_df, pivot_df, corr_df, price_df = run_backtest()

        return {
            "status": "success",
            "summary": summary_df.to_dict(orient="records"),
            "pivot": pivot_df.reset_index().to_dict(orient="records"),
            "correlation": corr_df.to_dict(),
            "price_log": price_df.to_dict(orient="records"),
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )
