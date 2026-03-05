import math
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

def lead_lag_analysis(
    tier1_stock,
    tier2_stocks,
    max_lag=60
):
    """
    Lead-lag signal discovery between one Tier-1 stock and multiple Tier-2 stocks.

    For each return horizon, the function:
    1. Aligns Tier-1 and Tier-2 return time series
    2. Searches over lag windows [-max_lag, +max_lag]
    3. Finds the lag that maximizes Pearson correlation
    4. Ranks Tier-2 stocks by correlation strength
    """

    # Return horizons (different aggregation windows of returns)
    return_windows = [
        "intraday_returns",
        "two_week_returns",
        "one_month_returns",
        "three_month_returns",
        "six_month_returns",
        "one_year_returns"
    ]

    # Dictionary that will store final results
    # Format:
    # {
    #   return_window : [
    #       (tier2_stock_ticker, best_lag, best_correlation),
    #       ...
    #   ]
    # }
    results = {}
    print("HERE")

    # Loop through each return horizon separately
    for window in return_windows:

        # Dynamically access stock return dataframe attribute
        tier1_df = getattr(tier1_stock, window)

        # Skip this window if Tier-1 data is missing
        if tier1_df is None:
            continue

        # Extract Tier-1 return series
        tier1_series = tier1_df["Return"]

        # Store results for this horizon
        window_results = []

        # Compare Tier-1 stock against every Tier-2 stock
        for t2_stock in tier2_stocks:

            # Access Tier-2 return dataframe for the same horizon
            tier2_df = getattr(t2_stock, window)

            # Skip if Tier-2 data is missing
            if tier2_df is None:
                continue

            # Extract Tier-2 return series
            tier2_series = tier2_df["Return"]

            # Align time indices by keeping only overlapping timestamps
            # This prevents correlation distortion from misaligned data
            df = pd.concat(
                [tier1_series, tier2_series],
                axis=1,
                join="inner"
            ).dropna()

            # Require minimum sample size for stability
            if df.shape[0] < 20:
                continue

            # Split aligned dataframe into two series
            # s1 = Tier-1 returns
            # s2 = Tier-2 returns
            s1 = df.iloc[:, 0]
            s2 = df.iloc[:, 1]

            # Initialize best signal trackers
            # We want the lag that produces the strongest correlation
            best_corr = -1
            best_lag = 0

            # Search across lag window [-max_lag, +max_lag]
            # This brute-force search identifies optimal temporal alignment
            for lag in range(-max_lag, max_lag + 1):

                # Shift Tier-2 series in time
                # Positive lag shifts series forward
                # Negative lag shifts series backward
                shifted = s2.shift(lag)

                # Compute Pearson correlation between Tier-1 series and shifted Tier-2 series
                corr = s1.corr(shifted)

                # Skip invalid correlation values
                if corr is None or math.isnan(corr):
                    continue

                # Keep track of highest correlation and corresponding lag
                if corr > best_corr:
                    best_corr = corr
                    best_lag = lag

            # Store best signal information for this Tier-2 stock
            window_results.append(
                (
                    t2_stock.ticker,
                    best_lag,
                    best_corr
                )
            )

        # Sort Tier-2 stocks by correlation strength (strongest signal first)
        window_results.sort(key=lambda x: x[2], reverse=True)

        # Save results for this return horizon
        results[window] = window_results

    # Return complete lead-lag analysis results
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