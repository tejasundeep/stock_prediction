# Univariate lstm example
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

import yfinance as yf #Import yfinance library

#Get stock historical data
stock_data = yf.Ticker("TCS.NS").history(period="max")
stock_data = stock_data[:-2]
#print(stock_data)

# Get last n days data
get_last_thirty_days_data = stock_data[-30:]

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