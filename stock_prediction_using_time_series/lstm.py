# Import required libraries
import numpy as np
import yfinance as yf
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras import backend

# Get stock historical data
stock_data = yf.Ticker("TCS.NS").history(period="max")
stock_data = stock_data[:-8]
print(stock_data)

# Get last n days data
get_last_thirty_days_data = stock_data[-30:]

# Split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    x_column, y_column = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        x_column.append(seq_x)
        y_column.append(seq_y)

    return np.array(x_column), np.array(y_column)

# Define input sequence
raw_seq = np.array(stock_data["Close"])

# Choose a number of time steps
n_steps = len(get_last_thirty_days_data)

# Split into samples
x_column, y_column = split_sequence(raw_seq, n_steps)

# Reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
x_column = x_column.reshape((x_column.shape[0], x_column.shape[1], n_features))

# Define model
model = Sequential()
model.add(LSTM(50, activation="relu", input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

# Fit model
model.fit(x_column, y_column, epochs=200, verbose=1)

# Demonstrate prediction
x_input = np.array([get_last_thirty_days_data["Close"]])
x_input = x_input.reshape((1, n_steps, n_features))
predicted_y = model.predict(x_input, verbose=0)
print(predicted_y)

# Clear session
backend.clear_session()
