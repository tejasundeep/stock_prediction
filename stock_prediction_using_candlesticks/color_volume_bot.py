# Download market data from Yahoo! Finance's API
import yfinance as yf

# See the yahoo finance ticker for your stock symbol
stock_symbol = "AAPL"  # input("Enter name: ")

# Last 10 years data with interval of 1 day
api_data = yf.download(tickers=stock_symbol, period="10y", interval="1d")


def find_color(open, close):
    if open > close:
        return "red"
    elif close > open:
        return "green"
    else:
        return "black"


def find_volume(volume_current, volume_previous):
    if volume_current > volume_previous:
        return "high"
    elif volume_current < volume_previous:
        return "low"
    else:
        return "same"


def find_pattern(color, volume):
    color_range = ["green", "red"]
    volume_range = ["high", "low", "same"]

    color = color
    volume = volume

    if color in color_range and volume in volume_range:
        return f"{color}_volume_{volume}"
    else:
        return f"black_volume_{volume}"


# Candles Template
def candles_history():
    # Get candles history
    history = []

    for i in range(len(api_data)):
        history.append(
            find_pattern(
                find_color(api_data["Open"][i], api_data["Close"][i]),
                find_volume(api_data["Volume"][i], api_data["Volume"][i - 1]),
            )
        )

    history = history[:-1]
    return history


# Search Pattern Template
def search_pattern(previous_candle):
    return candles_history()[-previous_candle:]


# Predict candle template
def predict(candles_history, search_pattern):
    result = [
        [i, i + len(search_pattern)]
        for i in range(len(candles_history))
        if candles_history[i : i + len(search_pattern)] == search_pattern
    ]  # Get occurances of a series in a list

    next_candles = []  # Get the next candle after many occurance

    for i in range(len(result)):
        if len(candles_history) > result[i][1]:
            next_candles.append(candles_history[result[i][1]])

    if len(next_candles) > 0:
        return max(
            set(next_candles), key=next_candles.count
        )  # Find the most common element in a list
    else:
        return "Unable to predict!"


print(predict(candles_history(), search_pattern(1)))
