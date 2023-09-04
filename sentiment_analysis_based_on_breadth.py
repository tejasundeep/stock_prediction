import yfinance as yf
import pandas as pd

# Define the tickers for the stocks in the index (S&P 500 in this example)
# Note: This is just a sample; you'd typically use all 500 tickers
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']

# Download historical data
data = yf.download(tickers, start='2023-01-01', end='2023-09-02', group_by='ticker')

# Initialize Advance-Decline Line
ad_line = 0

# Calculate Advance-Decline Line for the latest trading day
for ticker in tickers:
    latest_close = data[ticker]['Close'][-1]
    prev_close = data[ticker]['Close'][-2]
    
    if latest_close > prev_close:
        ad_line += 1  # Advance
    elif latest_close < prev_close:
        ad_line -= 1  # Decline

# Determine market sentiment
if ad_line > 0:
    sentiment = 'Bullish'
elif ad_line < 0:
    sentiment = 'Bearish'
else:
    sentiment = 'Neutral'

print(f"Advance-Decline Line: {ad_line}")
print(f"Market Sentiment: {sentiment}")
