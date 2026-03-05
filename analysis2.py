import math
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

# Computes lead-lag relationships between a Tier 1 stock and multiple Tier 2 stocks across various return horizons.
def lead_lag_analysis(
    tier1_stock,
    tier2_stocks,
    significance_level=0.05,
    enforce_stationarity=True
):
    """
    Enhanced lead-lag signal discovery.

    Enhancements:
    - Dynamic minimum lag window by return horizon
    - T-statistic + p-value for correlation
    - Augmented Dickey-Fuller stationarity test
    - Optional filtering by statistical significance
    """

    return_windows = [
        "intraday_returns",
        "two_week_returns",
        "one_month_returns",
        "three_month_returns",
        "six_month_returns",
        "one_year_returns"
    ]

    # Minimum lag rules based on overlapping return logic
    min_lag_map = {
        "intraday_returns": 30,
        "two_week_returns": 30,
        "one_month_returns": 30,
        "three_month_returns": 30,
        "six_month_returns": 60,
        "one_year_returns": 90
    }

    results = {}

    for window in return_windows:

        tier1_df = getattr(tier1_stock, window)
        if tier1_df is None:
            continue

        tier1_series = tier1_df["Return"]

        # Determine required lag window
        effective_max_lag = min_lag_map.get(window, 30) # default to 30 if not specified

        window_results = []

        tier1_stationarity = stationary(tier1_df, significance_level)
        if enforce_stationarity and not tier1_stationarity:
            continue

        for t2_stock in tier2_stocks:

            tier2_df = getattr(t2_stock, window)
            if tier2_df is None:
                continue
            
            tier2_stationarity = stationary(tier2_df, significance_level)
            # Optional stationarity check
            if enforce_stationarity and not tier2_stationarity:
                continue

            tier2_series = tier2_df["Return"]

            df = pd.concat(
                [tier1_series, tier2_series],
                axis=1,
                join="inner"
            ).dropna()

            if df.shape[0] < 20:
                continue

            s1 = df.iloc[:, 0]
            s2 = df.iloc[:, 1]

            best_corr = -1
            best_lag = 0
            best_n = 0

            for lag in range(-effective_max_lag, effective_max_lag + 1):

                shifted = s2.shift(lag)
                aligned = pd.concat([s1, shifted], axis=1).dropna()

                if aligned.shape[0] < 20:
                    continue

                corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])

                if corr is None or math.isnan(corr):
                    continue

                if corr > best_corr:
                    best_corr = corr
                    best_lag = lag
                    best_n = aligned.shape[0]

            if best_corr == -1:
                continue

            # Compute t-stat and p-value
            t_stat, p_value = tstat(best_corr, best_n)

            # Store all statistical metrics
            window_results.append(
                {
                    "ticker": t2_stock.ticker,
                    "best_lag": best_lag,
                    "correlation": best_corr,
                    "t_stat": t_stat,
                    "p_value": p_value,
                    "n_obs": best_n,
                    "significant": p_value < significance_level
                }
            )

        # Sort by correlation strength
        window_results.sort(
            key=lambda x: x["correlation"],
            reverse=True
        )

        results[window] = window_results

    return results

def stationary(return_df, significance = 0.05):
    """
    Augmented Dickey-Fuller test.
    H0: series has a unit root (non-stationary)
    Reject H0 (stationary) if p-value < significance level.
    """
    if len(return_df) < 20:
        return False
    adf_result = adfuller(return_df["Return"].dropna(), autolag="AIC")
    p_value = adf_result[1]
    return p_value < significance


def tstat(r, n):
    """
    T-statistic for a Pearson correlation coefficient.
    Degrees of freedom = n - 2.
    Returns (t_stat, two-tailed p_value).
    """
    if n <= 2 or abs(r) >= 1.0:
        return float("nan"), float("nan")

    t_stat = r * math.sqrt(n - 2) / math.sqrt(1 - r ** 2)
    p_value = 2 * stats.t.sf(abs(t_stat), df=n - 2)
    return t_stat, p_value