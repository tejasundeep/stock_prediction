import numpy as np
import pandas as pd
import tensorflow as tf
import yfinance as yf
import gym
from gym import spaces
from keras.models import Model
from keras.layers import Dense, Input, Concatenate
from keras.optimizers import Adam

# Function to normalize data
def normalize_data(df):
    return (df - df.mean()) / df.std()

# Function to add technical indicators as features
def add_technical_indicators(df):
    df['Change'] = df['Close'].diff()
    df['Volatility'] = df['Change'].rolling(window=5).std()
    df['Momentum'] = df['Close'] - df['Close'].shift(5)
    return df.dropna()

# Custom Stock Trading Environment
class StockTradingEnv(gym.Env):
    def __init__(self, symbol='AAPL', period='1y'):
        super(StockTradingEnv, self).__init__()
        try:
            stock_data = yf.download(symbol, period=period)
            self.stock_data = normalize_data(stock_data[['Open', 'High', 'Low', 'Close', 'Volume']])
            self.stock_data = add_technical_indicators(self.stock_data)
        except Exception as e:
            print(f"Error downloading stock data: {e}")
            raise e
        self.current_step = 0
        self.max_steps = len(self.stock_data) - 1
        self.action_space = spaces.Box(low=np.array([0]), high=np.array([1]), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        return self._next_observation()

    def _next_observation(self):
        obs = self.stock_data.iloc[self.current_step][['Open', 'High', 'Low', 'Close', 'Volume', 'Volatility', 'Momentum']].values
        return obs

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.max_steps
        current_price = self.stock_data['Close'].iloc[self.current_step - 1]
        next_day_price = self.stock_data['Close'].iloc[self.current_step] if not done else current_price

        # Improved reward function
        price_change = next_day_price - current_price
        reward = np.sign(price_change) * np.abs(price_change)

        return self._next_observation(), reward, done, {}

# SAC actor network
class SACActor(Model):
    def __init__(self, state_dim):
        super(SACActor, self).__init__()
        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(128, activation='relu')
        self.mean = Dense(1, activation='tanh')

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        mean = self.mean(x)
        return mean

# SAC critic network
class SACCritic(Model):
    def __init__(self, state_dim):
        super(SACCritic, self).__init__()
        self.fc1 = Dense(128, activation='relu')
        self.fc2 = Dense(128, activation='relu')
        self.q_value = Dense(1)

    def call(self, state, action):
        state_action = Concatenate(axis=-1)([state, action])
        x = self.fc1(state_action)
        x = self.fc2(x)
        q_value = self.q_value(x)
        return q_value

# SAC agent
class SACAgent:
    def __init__(self, state_dim, actor_lr=0.001, critic_lr=0.001, alpha=0.2):
        self.actor = SACActor(state_dim)
        self.actor_optimizer = Adam(actor_lr)
        self.critic = SACCritic(state_dim)
        self.critic_optimizer = Adam(critic_lr)
        self.alpha = alpha

    def train(self, state, action, reward, next_state, done):
        # Reshape the state and next_state tensors
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        state = tf.expand_dims(state, axis=0)
        action = tf.convert_to_tensor(action, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        next_state = tf.expand_dims(next_state, axis=0)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            predicted_action = self.actor(state)
            q_value = self.critic(state, predicted_action)
            q_next_state_value = self.critic(next_state, predicted_action) if not done else 0

            actor_loss = -tf.reduce_mean(q_value)
            critic_loss = tf.reduce_mean(tf.square(reward + self.alpha * q_next_state_value - q_value))

        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)

        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

    def select_action(self, state):
        # Reshape the state tensor
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        state = tf.expand_dims(state, axis=0)
        action = self.actor(state)
        return np.squeeze(action.numpy())

# Initialize the stock trading environment
env = StockTradingEnv(symbol='AAPL', period='30d')
state_dim = env.observation_space.shape[0]
sac_agent = SACAgent(state_dim)

# Training loop
num_episodes = 100
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = sac_agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        sac_agent.train(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward

    print(f"Episode: {episode}, Total Reward: {total_reward}")

# After training, predict the next action
state = env.reset()
predicted_action = sac_agent.select_action(state)
print("Predicted action for the next trading day:", "Buy" if predicted_action > 0.5 else "Sell")
