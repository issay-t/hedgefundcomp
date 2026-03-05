import pandas as pd
import os
import yfinance as yf
import numpy as np
from datetime import timedelta

# get_closing_prices fetches historical daily closing prices from yfinance API (max period)
# and turns it into a pandas dataframe.
def get_closing_prices(symbol, use_cache=True):
    filename = os.path.join("cached_data", f"{symbol}_data_yf.pkl")
    raw_data = None
    if use_cache and os.path.exists(filename):
        print(f"Loading {symbol} historical yf prices from cache...")
        raw_data = pd.read_pickle(filename)
    else:
        try:
            print(f"Fetching {symbol} historical yf prices from yfinance API...")
            raw_data = yf.download(
                symbol,
                period="max",
                interval="1d",
                auto_adjust=True
            )
            raw_data.to_pickle(filename)
        except Exception as e:
            print("Failed to fetch data from yfinance:", e)
    
    if raw_data.empty:
        raise ValueError(f"No historical price data available for {symbol}.")
    
    # Clean data and change to universal format
    df = raw_data.copy()
    df.index = pd.to_datetime(df.index, format="%Y-%m-%d") # convert into date objects
    df = df.sort_index()
    df = df.droplevel(1, axis=1)  # remove second level which is ticker symbol
    df.index.name = "date"
    df.columns.name = None
    df = df[["Close"]] # Filter for only close prices
    print(df)
    return df