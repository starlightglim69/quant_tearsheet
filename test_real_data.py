"""
Test the tearsheet stats engine against real market data:
AAPL as the portfolio, SPY as the benchmark.
"""

import os
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from tearsheet_stats import (
    cagr, annualized_volatility, sharpe_ratio, max_drawdown,
    sortino_ratio, beta, alpha, calmar_ratio, value_at_risk,
)

api_key = os.environ["ALPACA_API_KEY"]
secret_key = os.environ["ALPACA_SECRET_KEY"]
client = StockHistoricalDataClient(api_key, secret_key)


def get_daily_returns(symbol):
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start="2024-01-01",
    )
    bars = client.get_stock_bars(request)
    df = bars.df
    if "symbol" in df.index.names:
        df = df.reset_index(level="symbol", drop=True)
    return df["close"].pct_change().dropna()


aapl_returns = get_daily_returns("AAPL")
spy_returns = get_daily_returns("SPY")

print(f"AAPL: {len(aapl_returns)} days of returns")
print(f"SPY:  {len(spy_returns)} days of returns")

print("\n=== AAPL Tearsheet (vs SPY benchmark) ===")
print(f"CAGR:              {cagr(aapl_returns)*100:.2f}%")
print(f"Volatility:        {annualized_volatility(aapl_returns)*100:.2f}%")
print(f"Sharpe Ratio:      {sharpe_ratio(aapl_returns):.2f}")
print(f"Sortino Ratio:     {sortino_ratio(aapl_returns):.2f}")
print(f"Max Drawdown:      {max_drawdown(aapl_returns)*100:.2f}%")
print(f"Calmar Ratio:      {calmar_ratio(aapl_returns):.2f}")
print(f"95% VaR (daily):   {value_at_risk(aapl_returns)*100:.2f}%")
print(f"Beta (vs SPY):     {beta(aapl_returns, spy_returns):.3f}")
print(f"Alpha (vs SPY):    {alpha(aapl_returns, spy_returns)*100:.2f}%")
