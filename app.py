"""
Quant Tearsheet -- upload a CSV of daily returns, get a full
professional-grade performance and risk breakdown.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tearsheet_stats import (
    cagr, annualized_volatility, sharpe_ratio, max_drawdown,
    sortino_ratio, beta, alpha, alpha_significance, calmar_ratio, value_at_risk,
    TRADING_DAYS_PER_YEAR,
)


st.set_page_config(layout="wide")
st.title("Quant Tearsheet")
st.write("Upload your daily portfolio returns and get a complete performance & risk breakdown vs. a benchmark.")

st.subheader("Expected CSV format")
st.code("date,portfolio_return,benchmark_return\n2024-01-02,0.0012,0.0009\n2024-01-03,-0.0034,-0.0021")

uploaded_file = st.file_uploader("Drag & drop your CSV here", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
    data.columns = [c.lower() for c in data.columns]

    col_map = {}
    for col in data.columns:
        if "portfolio" in col:
            col_map["portfolio"] = col
        elif "benchmark" in col or "spy" in col:
            col_map["benchmark"] = col

    if "portfolio" not in col_map:
        st.error("Could not find a portfolio return column. Expected a column with 'portfolio' in its name.")
        st.stop()

    portfolio_returns = data[col_map["portfolio"]].dropna()
    has_benchmark = "benchmark" in col_map
    if has_benchmark:
        benchmark_returns = data[col_map["benchmark"]].dropna()

    st.success(f"Loaded {len(portfolio_returns)} daily observations from {portfolio_returns.index.min().date()} to {portfolio_returns.index.max().date()}")

    st.subheader("Portfolio Tearsheet")

    metrics_data = {
        "CAGR": (cagr(portfolio_returns), "pct", "Annualized growth rate"),
        "Volatility": (annualized_volatility(portfolio_returns), "pct", "Annualized standard deviation"),
        "Sharpe Ratio": (sharpe_ratio(portfolio_returns), "num", "Risk-adjusted return"),
        "Sortino Ratio": (sortino_ratio(portfolio_returns), "num", "Downside risk-adjusted return"),
        "Max Drawdown": (max_drawdown(portfolio_returns), "pct", "Worst peak-to-trough decline"),
        "Calmar Ratio": (calmar_ratio(portfolio_returns), "num", "CAGR / max drawdown"),
        "95% VaR (daily)": (value_at_risk(portfolio_returns), "pct", "Typical worst-day threshold"),
    }

    if has_benchmark:
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        aligned.columns = ["portfolio", "benchmark"]
        b = beta(aligned["portfolio"], aligned["benchmark"])
        a = alpha(aligned["portfolio"], aligned["benchmark"])
        sig = alpha_significance(aligned["portfolio"], aligned["benchmark"])
        metrics_data["Beta"] = (b, "num", "Market exposure vs benchmark")
        metrics_data["Alpha"] = (a, "pct", "Annualized return not explained by beta")

    cols = st.columns(4)
    for i, (label, (value, fmt, explanation)) in enumerate(metrics_data.items()):
        with cols[i % 4]:
            display_value = f"{value*100:.2f}%" if fmt == "pct" else f"{value:.2f}"
            st.metric(label, display_value, help=explanation)

    if has_benchmark:
        if sig["significant"]:
            st.info(f"✅ Alpha is statistically significant (p={sig['p_value']:.3f}) — this outperformance is unlikely to be due to chance alone.")
        else:
            st.warning(f"⚠️ Alpha is NOT statistically significant (p={sig['p_value']:.3f}) — this outperformance could plausibly be due to random chance rather than genuine skill.")

    st.subheader("Equity Curve - Growth of $1")

    portfolio_equity = (1 + portfolio_returns).cumprod()
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(portfolio_equity.index, portfolio_equity.values, label="Portfolio", color="#00cc96", linewidth=1.5)

    if has_benchmark:
        benchmark_equity = (1 + aligned["benchmark"]).cumprod()
        ax1.plot(benchmark_equity.index, benchmark_equity.values, label="Benchmark", color="#888888", linestyle="--", linewidth=1)

    ax1.set_ylabel("Growth of $1")
    ax1.legend()
    ax1.grid(alpha=0.2)
    st.pyplot(fig1)

    st.subheader("Underwater Plot — Drawdown from Peak")

    running_max = portfolio_equity.cummax()
    drawdown_series = (portfolio_equity / running_max - 1) * 100

    fig2, ax2 = plt.subplots(figsize=(14, 4))
    ax2.fill_between(drawdown_series.index, drawdown_series.values, 0, color="#ff4b4b", alpha=0.4)
    ax2.plot(drawdown_series.index, drawdown_series.values, color="#ff4b4b", linewidth=1)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(alpha=0.2)
    st.pyplot(fig2)