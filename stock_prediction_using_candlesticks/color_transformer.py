import yfinance as yf
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping

# Fetch Data
ticker = "AAPL"
data = yf.download(ticker, start="2020-01-01", end="2022-01-01")
data['Return'] = data['Adj Close'].pct_change()
data.dropna(inplace=True)
data['Color'] = np.where(data['Return'] > 0, 1, 0)

# Prepare Sequences
sequence_length = 10
n = len(data)
X, y = [], []
for i in range(n - sequence_length):
    X.append(data['Color'].values[i:i+sequence_length])
    y.append(data['Color'].values[i+sequence_length])

# Data normalization
scaler = MinMaxScaler()
X = np.array(X)
X = scaler.fit_transform(X)

y = np.array(y).reshape(-1, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define Transformer Model
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + tf.keras.layers.Dropout(0.2)(ffn_output))  # Added Dropout

# Model Architecture
embed_dim = 32
inputs = tf.keras.layers.Input(shape=(sequence_length,))
embedding_layer = tf.keras.layers.Embedding(input_dim=2, output_dim=embed_dim)(inputs)
transformer_block = TransformerBlock(embed_dim)
x = transformer_block(embedding_layer)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dropout(0.5)(x)  # Added Dropout
x = tf.keras.layers.Dense(32, activation="relu")(x)
x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
model = tf.keras.Model(inputs=inputs, outputs=x)

# Compile and Train Model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Early Stopping
early_stop = EarlyStopping(monitor='val_loss', patience=2)

# Model Training
model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stop])

# Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Model's test accuracy: {accuracy*100:.2f}%")

# Predict Next Day's Candle
recent_data = data['Color'].values[-sequence_length:]
recent_data = recent_data.reshape((1, sequence_length))
prediction = model.predict(recent_data)
next_candle_color = 'green' if prediction >= 0.5 else 'red'
print(f"Predicted color of the next day's candle: {next_candle_color}")
