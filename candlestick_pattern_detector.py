import yfinance as yf
import pandas as pd

# Download historical stock data
symbol = "AAPL"
data = yf.download(symbol, interval='5m', period="7d")

# Calculate candle body and wick sizes
data['Body'] = abs(data['Close'] - data['Open'])
data['Wick'] = data['High'] - data['Low'] - data['Body']

# Calculate median body and wick sizes
median_body = data['Body'].median()
median_wick = data['Wick'].median()

# Categorize each candle based on body size, wick size, and color
def categorize_candle(row):
    body_size = 'L' if row['Body'] >= median_body else 'S'
    wick_size = 'L' if row['Wick'] >= median_wick else 'S'
    color = 'G' if row['Close'] > row['Open'] else 'R'
    return f"{body_size}{color}{wick_size}W"

# Classify each candle
data['Category'] = data.apply(categorize_candle, axis=1)

# Initialize accuracy counts
accuracy_counts = {}

# Find patterns with high accuracy
for i in range(len(data) - 1):
    current_candle = data['Category'].iloc[i]
    next_move = data['Category'].iloc[i + 1]
    
    if current_candle not in accuracy_counts:
        accuracy_counts[current_candle] = {'Counts': {}, 'Total': 0}
    
    if next_move not in accuracy_counts[current_candle]['Counts']:
        accuracy_counts[current_candle]['Counts'][next_move] = 0
    
    accuracy_counts[current_candle]['Counts'][next_move] += 1
    accuracy_counts[current_candle]['Total'] += 1

high_accuracy_patterns = {}

# Calculate accuracy
for pattern, data in accuracy_counts.items():
    total = data['Total']
    for next_move, count in data['Counts'].items():
        accuracy = count / total
        if accuracy > 0.85:
            high_accuracy_patterns[f"{pattern}->{next_move}"] = {'Count': count, 'Accuracy': accuracy * 100}

# Print high accuracy patterns
print("High accuracy patterns:")
for pattern, stats in high_accuracy_patterns.items():
    print(f"Pattern: {pattern}, Count: {stats['Count']}, Accuracy: {stats['Accuracy']:.2f}%")
