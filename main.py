from universe import *
from ai_universe import *
from stock import *
from graph import *
from analysis2 import *

def main():
    tier1_stocks = {}
    tier2_stocks = {}
    for ticker in primary + suppliers:
        try:
            stock = Stock(ticker)
            if ticker in primary:
                tier1_stocks[ticker] = stock
            else:
                tier2_stocks[ticker] = stock
        except Exception as e:
            print(f"Error initializing stock {ticker}: {e}")

    # LEAD-LAG & CORRELATION ANALYSIS:
    # Note corr > 0.5 is considered strong
    for t1_ticker, t1_stock in tier1_stocks.items():
        print("====================================")
        print(f"\nLead-Lag Analysis for {t1_ticker}:")

        results = lead_lag_analysis(
            t1_stock,
            tier2_stocks.values(),
        )

        # First loop: return windows
        for window_name, window_results in results.items():

            print(f"\n--- {window_name} ---")

            # Second loop: tier2 results inside window
            for result in window_results:

                t2_ticker = result["ticker"]
                best_lag = result["best_lag"]
                best_corr = result["correlation"]
                t_stat = result["t_stat"]
                p_value = result["p_value"]
                n_obs = result["n_obs"]
                significant = result["significant"]

                print(
                    f"{t2_ticker} | "
                    f"Lag: {best_lag} | "
                    f"Corr: {best_corr:.4f} | "
                    f"T-stat: {t_stat:.2f} | "
                    f"P-value: {p_value:.4f} | "
                    f"N: {n_obs} | "
                    f"Significant: {significant}"
                )

        print("====================================")

    # GRAPHS:
    # Option to look at different return windows (intraday, 2 week, 1 month, 3 month, 6 month, 1 year)
    # - intraday_returns
    # - two_week_returns
    # - one_month_returns
    # - three_month_returns
    # - six_month_returns
    # - one_year_returns
    plot_tier1_vs_tier2(tier1_stocks, tier2_stocks, return_attr="one_month_returns")
    
if __name__ == "__main__":
    main()