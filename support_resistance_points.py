import yfinance as yf
import mplfinance as mpf
import pandas as pd
import numpy as np

def fetch_data(stock_symbol, period, interval):
    data = yf.download(stock_symbol, period=period, interval=interval)
    return data

def calculate_support_resistance(data):
    highs = data['High']
    lows = data['Low']

    from scipy.signal import argrelextrema
    order = 5
    support_indices = argrelextrema(lows.values, np.less, order=order)[0]
    resistance_indices = argrelextrema(highs.values, np.greater, order=order)[0]

    support_levels = pd.Series(lows[support_indices], index=lows.index[support_indices])
    resistance_levels = pd.Series(highs[resistance_indices], index=highs.index[resistance_indices])

    return support_levels, resistance_levels

def plot_candlestick_with_levels(data, support_levels, resistance_levels):
    data_subset = data[-180:]  # Last 180 days of data

    # Reindex to match the data_subset's index
    support_levels_subset = support_levels.reindex(data_subset.index)
    resistance_levels_subset = resistance_levels.reindex(data_subset.index)

    apds = [mpf.make_addplot(support_levels_subset, type='scatter', markersize=100, marker='^', color='green'),
            mpf.make_addplot(resistance_levels_subset, type='scatter', markersize=100, marker='v', color='red')]

    mpf.plot(data_subset, type='candle', style='charles', addplot=apds, title='Candlestick chart with Support and Resistance', volume=True)

# Example Usage
stock_symbol = 'AAPL'
period = '1y'
interval = '1d'

data = fetch_data(stock_symbol, period, interval)
support_levels, resistance_levels = calculate_support_resistance(data)
plot_candlestick_with_levels(data, support_levels, resistance_levels)
