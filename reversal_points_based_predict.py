import yfinance as yf
import numpy as np
from scipy.signal import argrelextrema
from sklearn.ensemble import RandomForestClassifier

# Fetch historical data
ticker = "INDIANB.NS"
data = yf.download(ticker, period="max", interval="1d")

# Convert the date index to a column
data.reset_index(inplace=True)

# Function to find local maxima and minima
def find_reversals(series, order=3):
    maxima = argrelextrema(series.values, np.greater, order=order)[0]
    minima = argrelextrema(series.values, np.less, order=order)[0]
    return maxima, minima

# Identify reversal points
maxima, minima = find_reversals(data['Close'])

# Collect data for classification
reversals = np.sort(np.concatenate([maxima, minima]))
reversal_days = data.index[reversals]
reversal_prices = data.iloc[reversals]['Close']

# Prepare dataset for classification
X = np.column_stack([np.diff(reversal_days), reversal_prices[:-1]])
y = np.array([1 if reversal_days[i] in maxima else 0 for i in range(len(reversal_days)-1)])

# Fit Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Predict the next reversal point
last_reversal_features = np.array([[reversal_days[-1] - reversal_days[-2], reversal_prices.iloc[-1]]])
next_reversal_type = model.predict(last_reversal_features)[0]

# Determine the type of the last reversal (bullish or bearish)
last_reversal_type = "Bearish" if y[-1] == 0 else "Bullish"

# Predict the price of the next reversal
next_reversal_price = reversal_prices.iloc[-1] if next_reversal_type == 1 else reversal_prices.iloc[-1]

print(f"Predicted Next Reversal Type: {'Bullish' if next_reversal_type == 1 else 'Bearish'} (Last Reversal was {last_reversal_type})")
print(f"Predicted Next Reversal Price: {next_reversal_price}")
