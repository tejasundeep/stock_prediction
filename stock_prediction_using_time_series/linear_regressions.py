import yfinance as yf # Import yfinance library for working with financial data
import numpy as np # Import numpy for array manipulation
from sklearn.model_selection import train_test_split # Import train_test_split for creating training and testing datasets
from sklearn.linear_model import LinearRegression # Import LinearRegression for building the model

# Get stock historical data using the yfinance library
df = yf.Ticker("TCS.NS").history(period="max")

# set cut off for stock history
stock_history_cut_off = 1

#last dataframe
last_df = df[:-(stock_history_cut_off-1)]
df=df[:-stock_history_cut_off]
print(last_df)

# Convert single dimension array to multi-dimensional array
# Use the open price as features for the model
x_column = np.array(df["Open"]).reshape(-1, 1)

# Use the low price as labels for the model
y_column = np.array(df["Low"]).reshape(-1, 1)  

# Splitting the dataset into train and test parts
x_train, x_test, y_train, y_test = train_test_split(
    x_column,
    y_column,
    test_size=0.3,  # 10% of the data will be used as test
    train_size=0.7,  # 70% of the data will be used as training
)

# Predict using the last available data point as input
input = np.array([last_df.tail(1)['Open']]).reshape(-1, 1)

# Create Linear Regression model
model = LinearRegression()
# Train the model using the training dataset
model.fit(x_train, y_train)

# Make a prediction using the model
prediction = model.predict(input)

# Print the prediction
print(prediction)