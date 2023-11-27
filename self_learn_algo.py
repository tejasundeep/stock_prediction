import yfinance as yf
import random
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import Adam, RMSprop, Nadam
import keras.backend as K
from keras.regularizers import l1, l2
from keras.callbacks import LearningRateScheduler

# Fetch stock data, including trading volume
def fetch_stock_data(symbol, period):
    stock_data = yf.download(symbol, period=period)
    close_prices = stock_data['Close'].tolist()
    volumes = stock_data['Volume'].tolist()
    return close_prices, volumes

# Function to calculate Sharpe Ratio
def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0

# Get the state, including price change, volume, and daily returns
def get_state_with_returns(data, volume, t, n, history_length=10):
    d = t - history_length + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]
    price_change = np.array([block[i + 1] - block[i] for i in range(history_length - 1)])
    price_change = (price_change - np.mean(price_change)) / np.std(price_change)

    volume_block = volume[d:t + 1] if d >= 0 else -d * [volume[0]] + volume[0:t + 1]
    normalized_volume = (np.array(volume_block) - np.mean(volume_block)) / np.std(volume_block)

    combined_state = np.concatenate((price_change, normalized_volume[-1:]))
    daily_returns = np.array([data[i + 1] / data[i] - 1 for i in range(len(data) - 1)])
    daily_returns = np.append(daily_returns, 0)  # Append a zero for the last day

    return np.reshape(combined_state, [1, len(combined_state)]), daily_returns

# Time-based decay for dynamic learning rate adjustment
def time_based_decay(epoch, lr):
    decay = 0.0001
    return lr / (1 + decay * epoch)

# Build the DQN model with dynamic learning rate and optimizer selection
def build_model(input_shape, action_space, optimizer_type='Adam', lr_schedule=None, l1_reg=0.01, l2_reg=0.01):
    inputs = Input(shape=input_shape)
    x = Dense(64, activation="relu", kernel_regularizer=l2(l2_reg))(inputs)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu", kernel_regularizer=l1(l1_reg))(x)
    x = Dropout(0.2)(x)

    state_value = Dense(1)(x)
    raw_advantage = Dense(action_space)(x)

    advantage = Lambda(lambda x: x - K.mean(x, axis=1, keepdims=True))(raw_advantage)
    q_value = Lambda(lambda x: x[0] + x[1])([state_value, advantage])

    if optimizer_type == 'RMSprop':
        optimizer = RMSprop(learning_rate=0.001 if lr_schedule is None else lr_schedule)
    elif optimizer_type == 'Nadam':
        optimizer = Nadam(learning_rate=0.002 if lr_schedule is None else lr_schedule)
    else:
        optimizer = Adam(learning_rate=0.005 if lr_schedule is None else lr_schedule)

    model = Model(inputs=inputs, outputs=q_value)
    model.compile(loss="mse", optimizer=optimizer)

    return model

# Hindsight Experience Replay (HER)
def hindsight_experience_replay(model, state, next_state, data, t, gamma, multi_step):
    alternative_action = 1 - np.argmax(model.predict(state)[0])
    hindsight_reward = np.sum([data[t + i + 1] - data[t + i] if alternative_action == 0 else data[t + i] - data[t + i + 1] for i in range(multi_step)])

    hindsight_target = hindsight_reward + gamma * np.amax(model.predict(next_state)[0])
    hindsight_target_f = model.predict(state)
    hindsight_target_f[0][alternative_action] = hindsight_target

    model.fit(state, hindsight_target_f, epochs=1, verbose=0)

# Penalty parameters
VOLUME_PENALTY_THRESHOLD = 0.1  # 10% of average volume
FREQUENT_TRADING_PENALTY = -0.5  # Penalty for each trade action
MAX_DRAWDOWN_PENALTY = 0.1  # Penalty for maximum drawdown
CONSISTENCY_REWARD = 0.2  # Reward for consistency in positive returns

# Function to calculate penalty for unrealistic trade volumes
def volume_penalty(trade_volume, average_volume):
    if trade_volume > VOLUME_PENALTY_THRESHOLD * average_volume:
        return -np.abs(trade_volume - VOLUME_PENALTY_THRESHOLD * average_volume) / average_volume
    return 0

# Function to calculate frequent trading penalty
def frequent_trading_penalty(num_trades, total_timesteps):
    return FREQUENT_TRADING_PENALTY * num_trades / total_timesteps

# Fetch data and initialize parameters
symbol = 'INDIANB.NS'
period = '30d'
data, volume = fetch_stock_data(symbol, period)

epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
gamma = 0.9
n = 5
history_length = 10  # The history length for state representation
multi_step = 3  # Define the number of steps to look ahead for the reward calculation
optimizer_type = 'Nadam'  # Can be 'Adam', 'RMSprop', or 'Nadam'
model = build_model((history_length,), 2, optimizer_type=optimizer_type)

# DQN Learning Loop
for episode in range(1, 101):
    total_reward = 0
    num_trades = 0
    state, daily_returns = get_state_with_returns(data, volume, 0, n, history_length)
    average_volume = np.mean(volume)

    # Initialize variables for risk management and consistency
    max_drawdown = 0
    current_portfolio_value = 0
    max_portfolio_value = 0
    consistent_days = 0
    max_consistent_days = 0

    for t in range(history_length, len(data) - multi_step):
        if np.random.rand() <= epsilon:
            action = random.choice([0, 1])
            num_trades += 1
        else:
            q_values = model.predict(state)
            action = np.argmax(q_values[0])

        next_state, next_daily_returns = get_state_with_returns(data, volume, t + multi_step, n, history_length)
        next_sharpe_ratio = calculate_sharpe_ratio(next_daily_returns[:t + multi_step])

        # Reward calculation
        reward = np.sum([data[t + i + 1] - data[t + i] for i in range(multi_step)])  # Multi-step reward
        risk_adjusted_reward = reward * next_sharpe_ratio  # New risk-adjusted reward
        trade_volume_penalty = volume_penalty(volume[t], average_volume)

        # Calculate daily portfolio value
        current_portfolio_value *= (1 + next_daily_returns[-1])

        # Update max drawdown
        if current_portfolio_value > max_portfolio_value:
            max_portfolio_value = current_portfolio_value
        else:
            drawdown = max_portfolio_value - current_portfolio_value
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Check for consistency in positive returns
        if next_daily_returns[-1] > 0:
            consistent_days += 1
            if consistent_days > max_consistent_days:
                max_consistent_days = consistent_days
        else:
            consistent_days = 0

        # Include risk management and consistency in the reward
        reward = risk_adjusted_reward - MAX_DRAWDOWN_PENALTY * max_drawdown + CONSISTENCY_REWARD * max_consistent_days

        total_reward += reward + trade_volume_penalty

        # Updating the target for the Q-network
        target = risk_adjusted_reward + gamma * np.amax(model.predict(next_state)[0])
        target_f = model.predict(state)
        target_f[0][action] = target

        # Apply dynamic learning rate
        lr_scheduler = LearningRateScheduler(time_based_decay)
        model.fit(state, target_f, epochs=1, verbose=0, callbacks=[lr_scheduler])

        state = next_state

        # Apply HER
        if episode % 20 == 0:  # Applying HER periodically
            hindsight_experience_replay(model, state, next_state, data, t, gamma, multi_step)

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

    frequent_trading_penalty_value = frequent_trading_penalty(num_trades, len(data) - history_length)
    total_reward += frequent_trading_penalty_value

    print(f'Episode: {episode}, Total Reward: {total_reward}, Number of Trades: {num_trades}')

# Predict action for the next day
state, _ = get_state_with_returns(data, volume, len(data) - 1, n, history_length)
q_values = model.predict(state)
predicted_action = 'Buy' if np.argmax(q_values[0]) == 0 else 'Sell'
print(f'Predicted action for the next day: {predicted_action}')
