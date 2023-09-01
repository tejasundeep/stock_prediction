import yfinance as yf
import pandas as pd
from typing import Dict

def download_data(symbol: str, interval: str = "1m", period: str = "1d") -> pd.DataFrame:
    """Download stock data using yfinance library."""
    try:
        return yf.download(symbol, interval=interval, period=period)
    except Exception as e:
        print(f"An error occurred while downloading data: {e}")
        return pd.DataFrame()

def build_pattern(window: pd.DataFrame) -> str:
    """Build pattern string based on window of stock data."""
    return "".join(
        "G" if row['Close'] > row['Open'] else "R"
        for _, row in window.iterrows()
    )

def calculate_pattern_accuracy(counts: Dict[str, int]) -> float:
    """Calculate accuracy of a particular pattern."""
    return max(counts['G'], counts['R']) / counts['Total']

def find_high_accuracy_patterns(data: pd.DataFrame, min_window_size: int, max_window_size: int, min_occurrences: int, accuracy_threshold: float) -> Dict[str, Dict]:
    """Find high-accuracy patterns in stock data."""
    if data.empty:
        print("No data to process.")
        return {}

    accuracy_counts = {}
    high_accuracy_patterns = {}

    for window_size in range(min_window_size, max_window_size + 1):
        for i in range(0, len(data) - window_size - 1):  
            window = data.iloc[i:i + window_size]
            pattern = build_pattern(window)

            next_candle = data.iloc[i + window_size]
            next_move = 'G' if next_candle['Close'] > next_candle['Open'] else 'R'

            accuracy_counts.setdefault(pattern, {'G': 0, 'R': 0, 'Total': 0})
            accuracy_counts[pattern][next_move] += 1
            accuracy_counts[pattern]['Total'] += 1

    for pattern, counts in accuracy_counts.items():
        if counts['Total'] >= min_occurrences:
            accuracy = calculate_pattern_accuracy(counts)
            if accuracy >= accuracy_threshold:
                high_accuracy_patterns[pattern] = {**counts, 'Accuracy': accuracy * 100}

    return high_accuracy_patterns

def predict_next_move(data: pd.DataFrame, high_accuracy_patterns: Dict[str, Dict], window_size: int) -> None:
    """Predict next move based on the last pattern and high-accuracy patterns."""
    last_window = data.iloc[-window_size:]
    last_pattern = build_pattern(last_window)
    pattern_data = high_accuracy_patterns.get(last_pattern, None)

    if pattern_data:
        likely_move = 'G' if pattern_data['G'] > pattern_data['R'] else 'R'
        print(f"Next likely move: {likely_move}, Accuracy: {pattern_data['Accuracy']:.2f}%")
    else:
        print("No pattern found for prediction.")

if __name__ == "__main__":
    symbol = "AAPL"
    interval = "1m"
    period = "1d"
    min_window_size = 2
    min_occurrences = 2
    accuracy_threshold = 0.9
    max_window_size = min_window_size ** 10

    data = download_data(symbol, interval, period)
    high_accuracy_patterns = find_high_accuracy_patterns(data, min_window_size, max_window_size, min_occurrences, accuracy_threshold)

    print("High accuracy patterns:")
    for pattern, counts in high_accuracy_patterns.items():
        print(f"Pattern: {pattern}, Counts: {counts}")

    print("Predicting the next move based on the last pattern:")
    predict_next_move(data, high_accuracy_patterns, min_window_size)
