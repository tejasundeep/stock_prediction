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


def top_wick(high, low, open, close):
    if open > close:
        return (high - open) / (high - low) * 100
    elif close > open:
        return (high - close) / (high - low) * 100
    elif high - close != 0 or high - open != 0 or high - low != 0:
        return (high - open) / (high - low) * 100
    else:
        return 0


def bottom_wick(high, low, open, close):
    if open > close:
        return (close - low) / (high - low) * 100
    elif close > open:
        return (open - low) / (high - low) * 100
    elif high - close != 0 or high - open != 0 or high - low != 0:
        return (close - low) / (high - low) * 100
    else:
        return 0


def body_size(high, low, open, close):
    if open > close:
        return (open - close) / (high - low) * 100
    elif close > open:
        return (close - open) / (high - low) * 100
    elif high - close != 0 or high - open != 0 or high - low != 0:
        return (open - close) / (high - low) * 100
    else:
        return 0


def find_pattern(color, top_wick, bottom_wick, body_size, volume):
    color_range = ["green", "red"]
    top_wick_range = list(range(101))
    bottom_wick_range = list(range(101))
    body_size_range = list(range(101))
    volume_range = ["high", "low", "same"]

    color = color
    top_wick = round(top_wick)
    bottom_wick = round(bottom_wick)
    body_size = round(body_size)
    volume = volume

    if (
        color in color_range
        and top_wick in top_wick_range
        and bottom_wick in bottom_wick_range
        and body_size in body_size_range
        and volume in volume_range
    ):
        return f"{color}_top_{top_wick}_bottom_{bottom_wick}_body_{body_size}_volume_{volume}"
    else:
        return f"black_top_{top_wick}_bottom_{bottom_wick}_body_{body_size}_volume_{volume}"


# Candles Template
def candles_history():
    # Get candles history
    history = []

    for i in range(len(api_data)):
        history.append(
            find_pattern(
                find_color(api_data["Open"][i], api_data["Close"][i]),
                top_wick(
                    api_data["High"][i],
                    api_data["Low"][i],
                    api_data["Open"][i],
                    api_data["Close"][i],
                ),
                bottom_wick(
                    api_data["High"][i],
                    api_data["Low"][i],
                    api_data["Open"][i],
                    api_data["Close"][i],
                ),
                body_size(
                    api_data["High"][i],
                    api_data["Low"][i],
                    api_data["Open"][i],
                    api_data["Close"][i],
                ),
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
