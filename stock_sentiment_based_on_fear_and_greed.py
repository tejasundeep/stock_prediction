import yfinance as yf
import numpy as np

def calculate_fear_and_greed_index(ticker):
    stock_data = yf.Ticker(ticker)
    hist_data = stock_data.history(period="1y")
    
    # Calculate momentum
    momentum = (hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[-125]) - 1
    daily_returns = hist_data['Close'].pct_change().dropna()
    
    # Calculate volatility
    volatility = np.std(daily_returns[-30:])
    
    # Calculate indices
    momentum_index = momentum * 50
    volatility_index = (1 - volatility) * 50
    
    # Calculate Fear and Greed Index
    fear_and_greed_index = (momentum_index + volatility_index) / 2
    return fear_and_greed_index

def get_market_sentiment(ticker):
    fear_and_greed = calculate_fear_and_greed_index(ticker)
    
    # Determine sentiment category
    if fear_and_greed <= 24:
        sentiment_category = "Extreme Fear"
    elif fear_and_greed <= 49:
        sentiment_category = "Fear"
    elif fear_and_greed == 50:
        sentiment_category = "Neutral"
    elif fear_and_greed <= 74:
        sentiment_category = "Greed"
    else:
        sentiment_category = "Extreme Greed"
    
    return {
        'Fear & Greed Index': fear_and_greed,
        'Composite Sentiment': fear_and_greed,
        'Sentiment Category': sentiment_category
    }

ticker = "AAPL"
sentiment_data = get_market_sentiment(ticker)
print(f"For {ticker}, the sentiment data is as follows:")
print(f"Sentiment Category: {sentiment_data['Sentiment Category']}")
