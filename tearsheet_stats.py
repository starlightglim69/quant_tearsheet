"""
Core statistics engine for the quant tearsheet -- pure functions, no
web framework dependencies, so every calculation can be tested and
verified independently before we ever build a UI around it.
 
All functions take a pandas Series of DAILY returns (as decimals,
e.g. 0.01 = +1%) and return a single number or another Series.
"""
 
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
 
TRADING_DAYS_PER_YEAR = 252
 
 
def cagr(returns: pd.Series) -> float:
    """Compound Annual Growth Rate -- the annualized rate of return."""
    equity_curve = (1 + returns).cumprod()
    # FIX: total_return should be measured from a starting value of 1,
    # not from equity_curve.iloc[0] (which is 1 + returns.iloc[0]).
    # Dividing by iloc[0] silently dropped day 0's return.
    total_return = equity_curve.iloc[-1] - 1
    n_years = len(returns) / TRADING_DAYS_PER_YEAR
    if n_years <= 0:
        return np.nan
    return (1 + total_return) ** (1 / n_years) - 1
 
 
def annualized_volatility(returns: pd.Series) -> float:
    """Annualized standard deviation of returns."""
    return returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
 
 
def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio: excess return per unit of volatility."""
    excess_daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess_returns = returns - excess_daily_rf
    if excess_returns.std() == 0:
        return np.nan
    return (excess_returns.mean() / excess_returns.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
 
 
def max_drawdown(returns: pd.Series) -> float:
    """Worst peak-to-trough decline."""
    equity_curve = (1 + returns).cumprod()
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return drawdown.min()
 
 
def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Like Sharpe, but only penalizes DOWNSIDE volatility."""
    excess_daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    excess_returns = returns - excess_daily_rf
 
    downside_returns = excess_returns[excess_returns < 0]
    # FIX: need at least 2 downside observations for .std() (ddof=1) to be
    # well-defined. With exactly 1, std() returns NaN and slips past the
    # `== 0` check below, silently propagating NaN upward.
    if len(downside_returns) < 2:
        return np.nan
    downside_std = downside_returns.std()
    if downside_std == 0:
        return np.nan
 
    return (excess_returns.mean() / downside_std) * np.sqrt(TRADING_DAYS_PER_YEAR)
 
 
def beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """How much the portfolio moves relative to the benchmark (e.g. SPY)."""
    covariance_matrix = np.cov(portfolio_returns, benchmark_returns)
    covariance = covariance_matrix[0, 1]
    benchmark_variance = covariance_matrix[1, 1]
    if benchmark_variance == 0:
        return np.nan
    return covariance / benchmark_variance
 
 
def alpha(portfolio_returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized CAPM alpha: return not explained by market exposure (beta)."""
    b = beta(portfolio_returns, benchmark_returns)
    portfolio_annual_return = cagr(portfolio_returns)
    benchmark_annual_return = cagr(benchmark_returns)
    expected_return = risk_free_rate + b * (benchmark_annual_return - risk_free_rate)
    return portfolio_annual_return - expected_return
 
 
def calmar_ratio(returns: pd.Series) -> float:
    """CAGR divided by max drawdown -- return per unit of worst-case pain."""
    mdd = max_drawdown(returns)
    if mdd == 0:
        return np.nan
    return cagr(returns) / abs(mdd)
 
 
def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical VaR: the loss level that returns are worse than only
    (1 - confidence) of the time. E.g. 95% VaR = the threshold that
    daily returns fall below only 5% of the time.
    """
    return returns.quantile(1 - confidence)
 
 
def alpha_significance(portfolio_returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate: float = 0.0) -> dict:
    """
    Tests whether alpha is statistically distinguishable from zero,
    using a t-test on the regression residuals (the daily 'unexplained'
    returns after removing the beta-driven component).
    """
    b = beta(portfolio_returns, benchmark_returns)
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
 
    expected_daily = daily_rf + b * (benchmark_returns - daily_rf)
    residuals = portfolio_returns - expected_daily
 
    n = len(residuals)
    mean_residual = residuals.mean()
    std_error = residuals.std() / np.sqrt(n)
 
    if std_error == 0:
        return {"t_stat": np.nan, "p_value": np.nan, "significant": False}
 
    t_stat = mean_residual / std_error
    p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df=n - 1))
 
    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "significant": p_value < 0.05,
    }
 