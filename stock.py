import copy

from fetch_data import *
from universe import *

# Max_lookback is the maximum number of historical data points to keep for returns calculations (default 3 years of trading days)
# note: greater than 3 years may lead to more noise and inflation of t-stats
class Stock:
    def __init__(self, ticker, max_lookback=252*3):
        self.ticker = ticker
        self.max_lookback = max_lookback

        self.intraday_returns = None
        self.two_week_returns = None
        self.one_month_returns = None
        self.three_month_returns = None
        self.six_month_returns = None
        self.one_year_returns = None

        self.close_prices = None

        self.close_prices = get_closing_prices(self.ticker)
        self.intraday_returns = self.calculate_returns(return_window=1)
        self.two_week_returns = self.calculate_returns(return_window=10)
        self.one_month_returns = self.calculate_returns(return_window=21)
        self.three_month_returns = self.calculate_returns(return_window=63)
        self.six_month_returns = self.calculate_returns(return_window=126)
        self.one_year_returns = self.calculate_returns(return_window=252)

    # calculate_returns computes the percentage change in closing prices to get returns.
    def calculate_returns(self, return_window=1):
        df = self.close_prices.copy()
        df["Return"] = df["Close"].pct_change(return_window)
        df = df[["Return"]].dropna()
        if len(df) > self.max_lookback:
            df = df.tail(self.max_lookback) # keep only the most recent max_lookback data points
        return df


        
