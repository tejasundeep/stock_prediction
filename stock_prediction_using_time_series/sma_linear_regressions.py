import yfinance as yf #Import yfinance library
import numpy as np #Import numpy library
from sklearn.linear_model import LinearRegression

#Get stock historical data
stock_data = yf.Ticker("TCS.NS").history(period="max")
stock_data = stock_data[:-2]
#print(stock_data)

# Get last n days data
get_last_thirty_days_data = stock_data[-30:]
get_last_fifteen_days_data = stock_data[-15:]
get_last_seven_days_data = stock_data[-7:]
get_last_five_days_data = stock_data[-5:]
get_last_three_days_data = stock_data[-3:]
get_last_one_day_data = stock_data[-1:]

# Get n Days sma
# 30 Days
thirty_days_open_sma = np.mean(get_last_thirty_days_data["Open"])
thirty_days_close_sma = np.mean(get_last_thirty_days_data["Close"])
thirty_days_high_sma = np.mean(get_last_thirty_days_data["High"])
thirty_days_low_sma = np.mean(get_last_thirty_days_data["Low"])
thirty_days_volume_sma = np.mean(get_last_thirty_days_data["Volume"])
# 15 Days
fifteen_days_open_sma = np.mean(get_last_fifteen_days_data["Open"])
fifteen_days_close_sma = np.mean(get_last_fifteen_days_data["Close"])
fifteen_days_high_sma = np.mean(get_last_fifteen_days_data["High"])
fifteen_days_low_sma = np.mean(get_last_fifteen_days_data["Low"])
fifteen_days_volume_sma = np.mean(get_last_fifteen_days_data["Volume"])
# 7 Days
seven_days_open_sma = np.mean(get_last_seven_days_data["Open"])
seven_days_close_sma = np.mean(get_last_seven_days_data["Close"])
seven_days_high_sma = np.mean(get_last_seven_days_data["High"])
seven_days_low_sma = np.mean(get_last_seven_days_data["Low"])
seven_days_volume_sma = np.mean(get_last_seven_days_data["Volume"])
# 5 Days
five_days_open_sma = np.mean(get_last_five_days_data["Open"])
five_days_close_sma = np.mean(get_last_five_days_data["Close"])
five_days_high_sma = np.mean(get_last_five_days_data["High"])
five_days_low_sma = np.mean(get_last_five_days_data["Low"])
five_days_volume_sma = np.mean(get_last_five_days_data["Volume"])
# 3 Days
three_days_open_sma = np.mean(get_last_three_days_data["Open"])
three_days_close_sma = np.mean(get_last_three_days_data["Close"])
three_days_high_sma = np.mean(get_last_three_days_data["High"])
three_days_low_sma = np.mean(get_last_three_days_data["Low"])
three_days_volume_sma = np.mean(get_last_three_days_data["Volume"])
# 1 Day
one_day_open_sma = np.mean(get_last_one_day_data["Open"])
one_day_close_sma = np.mean(get_last_one_day_data["Close"])
one_day_high_sma = np.mean(get_last_one_day_data["High"])
one_day_low_sma = np.mean(get_last_one_day_data["Low"])
one_day_volume_sma = np.mean(get_last_one_day_data["Volume"])

# Data
x_open = np.array([thirty_days_open_sma, fifteen_days_open_sma, seven_days_open_sma, five_days_open_sma, three_days_open_sma, one_day_open_sma])
y_close = np.array([thirty_days_low_sma, fifteen_days_low_sma, seven_days_low_sma, five_days_low_sma, three_days_low_sma, one_day_low_sma])

x_features = np.column_stack((x_open,))
y_features = np.column_stack((y_close,))

# Create and fit the model
model = LinearRegression()
model.fit(x_features, y_features)

# Predict next value in the series
next_val = model.predict([[3329.00]])
print(next_val)
