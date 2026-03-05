import matplotlib.pyplot as plt
import pandas as pd

def plot_tier1_vs_tier2(tier1_stocks, tier2_stocks, return_attr="one_month_returns"):
    for t1_ticker, t1_stock in tier1_stocks.items():
        
        # Collect all return series (Tier1 + all Tier2)
        series_dict = {}

        # Tier 1
        t1_returns = getattr(t1_stock, return_attr)["Return"]
        series_dict[t1_ticker] = t1_returns

        # Tier 2
        for t2_ticker, t2_stock in tier2_stocks.items():
            t2_returns = getattr(t2_stock, return_attr)["Return"]
            series_dict[t2_ticker] = t2_returns

        # ---- ALIGN TIMEFRAMES ----
        df = pd.concat(series_dict, axis=1, join="inner")

        # ---- PLOT ----
        plt.figure(figsize=(12, 6))

        # Tier 1
        plt.plot(df.index, df[t1_ticker], linewidth=3, label=t1_ticker)

        # Tier 2
        for t2_ticker in tier2_stocks:
            plt.plot(
                df.index,
                df[t2_ticker],
                linewidth=1,
                alpha=0.6,
                label=t2_ticker
            )

        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        plt.title(f"{return_attr} Overlay: {t1_ticker} vs Tier 2")
        plt.show()