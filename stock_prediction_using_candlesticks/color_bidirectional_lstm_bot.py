from numpy import array
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM
from keras.layers import Dense

# Download market data from Yahoo! Finance's API
import yfinance as yf

# See the yahoo finance ticker for your stock symbol
stock_symbol = "AAPL"  # input("Enter stock name: ")

# Last 10 years data with interval of 1 day
api_data = yf.download(tickers=stock_symbol, period="10y", interval="1d")


def find_color(open, close):
    if open > close:
        return 1
    elif close > open:
        return 2
    else:
        return 0


# Candles Template
def candle_history():
    # Get candles history
    history = []

    for i in range(len(api_data)):
        history.append(
            find_color(
                api_data["Open"][i],
                api_data["Close"][i],
            )
        )

    return history


# Split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    x_column, y_column = list(), list()
    for i in range(len(sequence)):
        # Find the end of this pattern
        end_ix = i + n_steps
        # Check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # Gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        x_column.append(seq_x)
        y_column.append(seq_y)

    return array(x_column), array(y_column)


# Define input sequence
raw_seq = candle_history()
raw_seq = raw_seq[:-2]  # Crop the candle for testing

short_seq_for_pred = []
for i in reversed(range(1, round((len(raw_seq) + 1) * 0.003))):
    short_seq_for_pred.append(raw_seq[-i:][0])

# Choose a number of time steps
n_steps = len(short_seq_for_pred)
print(short_seq_for_pred)

# Split into samples
x_column, y_column = split_sequence(raw_seq, n_steps)

# Reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
x_column = x_column.reshape((x_column.shape[0], x_column.shape[1], n_features))

# Define model
model = Sequential()
model.add(Bidirectional(LSTM(50, activation="relu"), input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])

# Fit model
model.fit(x_column, y_column, epochs=50, verbose=1)

# Demonstrate prediction
x_input = array(short_seq_for_pred)
x_input = x_input.reshape((1, n_steps, n_features))
predicted_y = model.predict(x_input, verbose=0)

if round(predicted_y[0][0]) == 1:
    print("Red Candle")
elif round(predicted_y[0][0]) == 2:
    print("Green Candle")
else:
    print("Black Candle")
