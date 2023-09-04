import yfinance as yf
import random
import numpy as np
from keras.models import Model
from keras.layers import Dense, Dropout, Input, Lambda
from keras.optimizers import Adam
import keras.backend as K

def fetch_stock_data(symbol, period):
    stock_data = yf.download(symbol, period=period)
    close_prices = stock_data['Close'].tolist()
    return close_prices

def get_state(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]
    res = np.array([block[i + 1] - block[i] for i in range(n - 1)])
    res = (res - np.mean(res)) / np.std(res)
    return np.reshape(res, [1, n - 1])

def build_model(input_shape, action_space):
    inputs = Input(shape=input_shape)
    x = Dense(64, activation="relu")(inputs)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.2)(x)
    
    state_value = Dense(1)(x)
    raw_advantage = Dense(action_space)(x)
    
    advantage = Lambda(lambda x: x - K.mean(x, axis=1, keepdims=True))(raw_advantage)
    q_value = Lambda(lambda x: x[0] + x[1])([state_value, advantage])
    
    model = Model(inputs=inputs, outputs=q_value)
    model.compile(loss="mse", optimizer=Adam(learning_rate=0.005))
    
    return model

# Hindsight Experience Replay
def hindsight_experience_replay(model, state, next_state, gamma):
    hindsight_action = 1 - np.argmax(model.predict(state)[0])
    hindsight_reward = 0

    if hindsight_action == 0:
        hindsight_reward = data[t + 1] - data[t]
    elif hindsight_action == 1:
        hindsight_reward = data[t] - data[t + 1]
    
    hindsight_target = hindsight_reward + gamma * np.amax(model.predict(next_state)[0])
    hindsight_target_f = model.predict(state)
    hindsight_target_f[0][hindsight_action] = hindsight_target
    
    model.fit(state, hindsight_target_f, epochs=1, verbose=0)
  
# Parameters
symbol = 'AAPL'
period = '1mo'
data = fetch_stock_data(symbol, period)
epsilon = 1.0
epsilon_decay = 0.995
min_epsilon = 0.01
gamma = 0.9
n = 5
hindsight_episode = random.randint(1, 100)

model = build_model((n - 1,), 2)

# DQN Learning
for episode in range(1, 101):
    print(f"Starting episode {episode}...")
    total_reward = 0
    state = get_state(data, 0, n)
    
    for t in range(n, len(data) - 1):
        print(f"Time step {t}...")
        
        if np.random.rand() <= epsilon:
            action = random.choice([0, 1])
        else:
            q_values = model.predict(state)
            action = np.argmax(q_values[0])
        
        next_state = get_state(data, t + 1, n)
        reward = 0
        
        if action == 0:
            reward = data[t + 1] - data[t]
        elif action == 1:
            reward = data[t] - data[t + 1]
        
        target = reward + gamma * np.amax(model.predict(next_state)[0])
        
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)
        
        state = next_state
        total_reward += reward
        
        # Apply HER
        if episode == hindsight_episode:
            hindsight_experience_replay(model, state, next_state, gamma)
        
    epsilon = max(min_epsilon, epsilon_decay * epsilon)
    print(f'Episode: {episode}, Total Reward: {total_reward}')

# Predict action for the next day
state = get_state(data, len(data) - 1, n)
q_values = model.predict(state)
predicted_action = 'Buy' if np.argmax(q_values[0]) == 0 else 'Sell'
print(f'Predicted action for the next day: {predicted_action}')
