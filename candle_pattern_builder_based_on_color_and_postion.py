import yfinance as yf
import pandas as pd

def download_data(symbol: str, interval: str = "1m", period: str = "7d") -> pd.DataFrame:
    try:
        return yf.download(symbol, interval=interval, period=period)
    except Exception as e:
        print(f"An error occurred while downloading data: {e}")
        return pd.DataFrame()

def build_pattern(window: pd.DataFrame) -> str:
    pattern_list = []
    for i in range(len(window)):
        if i == 0:
            pattern_list.append("G" if window['Close'].iloc[i] > window['Open'].iloc[i] else "R")
        else:
            if window['Close'].iloc[i] > window['Open'].iloc[i]:
                pattern = "G"
            else:
                pattern = "R"

            prev_close = window['Close'].iloc[i-1]
            current_open = window['Open'].iloc[i]
            current_close = window['Close'].iloc[i]

            if current_open > prev_close and current_close > prev_close:
                pattern += "A"
            elif current_open < prev_close and current_close < prev_close:
                pattern += "B"
            elif current_open > prev_close:
                pattern += "O"
            elif current_close > prev_close:
                pattern += "C"
            elif current_open < prev_close:
                pattern += "X"
            elif current_close < prev_close:
                pattern += "Y"

            pattern_list.append(pattern)

    return "".join(pattern_list)

def find_custom_patterns(data: pd.DataFrame, min_window_size: int, max_window_size: int, min_occurrences: int, accuracy_threshold: float) -> dict:
    if data.empty:
        print("No data to process.")
        return {}

    accuracy_counts = {}
    high_accuracy_patterns = {}

    for window_size in range(min_window_size, max_window_size + 1):
        for i in range(0, len(data) - window_size):
            window = data.iloc[i:i + window_size]
            pattern = build_pattern(window)
            next_move = 'U' if data['Close'].iloc[i + window_size] > window['Close'].iloc[-1] else 'D'

            accuracy_counts.setdefault(pattern, {'U': 0, 'D': 0, 'Total': 0})
            accuracy_counts[pattern][next_move] += 1
            accuracy_counts[pattern]['Total'] += 1

    for pattern, counts in accuracy_counts.items():
        if counts['Total'] >= min_occurrences:
            accuracy = max(counts['U'], counts['D']) / counts['Total']
            if accuracy >= accuracy_threshold:
                high_accuracy_patterns[pattern] = counts
                high_accuracy_patterns[pattern]['Accuracy'] = accuracy * 100

    return high_accuracy_patterns

def predict_next_move(data: pd.DataFrame, high_accuracy_patterns: dict, min_window_size: int) -> None:
    last_window = data.iloc[-min_window_size:]
    last_pattern = build_pattern(last_window)

    if last_pattern in high_accuracy_patterns:
        likely_move = 'U' if high_accuracy_patterns[last_pattern]['U'] > high_accuracy_patterns[last_pattern]['D'] else 'D'
        accuracy = high_accuracy_patterns[last_pattern]['Accuracy']
        print(f"Next likely move: {likely_move}, Accuracy: {accuracy}%")
    else:
        print("No pattern found.")

if __name__ == "__main__":
    symbol = "AAPL"
    data = download_data(symbol)
    min_window_size = 5
    max_window_size = min_window_size * 5
    min_occurrences = 5
    accuracy_threshold = 0.95

    high_accuracy_patterns = find_custom_patterns(data, min_window_size, max_window_size, min_occurrences, accuracy_threshold)

    print("High accuracy patterns:")
    for pattern, counts in high_accuracy_patterns.items():
        print(f"Pattern: {pattern}, Counts: {counts}, Accuracy: {counts['Accuracy']:.2f}%")

    print("Predicting the next move based on the last pattern:")
    predict_next_move(data, high_accuracy_patterns, min_window_size)
