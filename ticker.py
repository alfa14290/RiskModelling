import datetime as dt
import sys

import arch.data.sp500
import numpy as np
import pandas as pd
from arch import arch_model
import yfinance as yf
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def read(ticker):
    ticker = pdr.get_data_yahoo(ticker, start='2008-01-02', end='2020-12-31')

# Check first 3 and last 3 rows
    pd.concat([ticker.head(3), ticker.tail(3)])
    ticker = ticker[['Adj Close']]
    

# Plot
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(ticker, color='turquoise')
    ax.set(title='ticker', ylabel='Price per Share') 

    ax.axvline(pd.to_datetime('2020-11-30'), color='slategray', lw=1.2, linestyle='--')
    plt.show()
    
    ticker['return'] = ticker.pct_change().dropna() * 100

# Plot
    fig, ax = plt.subplots(figsize=(16,8))
    ax.plot(ticker['return'].dropna(), color='lightcoral')
    ax.set(title='ticker', ylabel='% Return') 

    ax.axvline(pd.to_datetime('2008-01-02'), color='slategray', lw=1.2, linestyle='--')
    plt.show()

#plot distribution looklike of return
    dist = ticker['return'].hist(bins=50)
    dist.set_xlabel('Return')
    dist.set_ylabel('Sample')
    dist.set_title('Return distribution looklike')
    plt.show()

    return ticker

r = read('FCX')




