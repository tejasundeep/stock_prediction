import yfinance as yf
from collections import Counter
import pandas as pd

# Function to fetch and preprocess data
def fetch_data(ticker, period):
    data = yf.Ticker(ticker).history(period=period)
    data['Color'] = ['Green' if close > open else 'Red' for open, close in zip(data['Open'], data['Close'])]
    average_volume = data['Volume'].mean()
    data['Volume Category'] = ['High' if volume > average_volume else 'Low' for volume in data['Volume']]
    return data

# Function to convert data to pattern strings
def convert_to_pattern(data):
    return [color[0] + volume[0] for color, volume in zip(data['Color'], data['Volume Category'])]

# Function to find occurrences of a pattern in the history
def find_pattern(pattern, history):
    matches = [i for i in range(len(history) - len(pattern) + 1) if history[i:i+len(pattern)] == pattern]
    return matches

# Function to predict the next day's pattern
def predict_next_day_pattern(ticker, period, pattern_length):
    data = fetch_data(ticker, period)
    pattern_data = convert_to_pattern(data)
    recent_pattern = pattern_data[-pattern_length:]
    history = pattern_data[:-pattern_length]
    matches = find_pattern(recent_pattern, history)
    next_day_patterns = [history[i + pattern_length] for i in matches if i + pattern_length < len(history)]
    counter = Counter(next_day_patterns)
    prediction = counter.most_common(1)[0][0] if next_day_patterns else 'No prediction (pattern not found)'
    return prediction

# Parameters
ticker = 'AAPL'
period = "1y"
pattern_length = 4

# Prediction
prediction = predict_next_day_pattern(ticker, period, pattern_length)
print(f'Predicted Pattern for next day: {prediction}')
