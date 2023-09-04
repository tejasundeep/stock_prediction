import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

def load_csv_data(file_path, column_name='Close'):
    try:
        df = pd.read_csv(file_path)
        if column_name not in df.columns:
            print(f"Column '{column_name}' not found in CSV.")
            return None
        df[column_name].fillna(method='ffill', inplace=True)
        return df[column_name].values
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None
    except pd.errors.EmptyDataError:
        print("CSV file is empty.")
        return None

def preprocess_time_series_data(data, sequence_length):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i+sequence_length])
        y.append(scaled_data[i+sequence_length])
    return np.array(X), np.array(y).reshape(-1, 1), scaler

def transformer_encoder_decoder(units, d_model, num_heads, dropout):
    inputs = tf.keras.Input(shape=(None, d_model))
    x = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + inputs)
    x = tf.keras.layers.Dense(units, activation='relu')(x)
    x = tf.keras.layers.Dense(d_model)(x)
    outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + inputs)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def stacked_transformer_encoder_decoder(units, d_model, num_heads, dropout, num_layers):
    inputs = tf.keras.Input(shape=(None, d_model))
    x = inputs
    for _ in range(num_layers):
        x = transformer_encoder_decoder(units, d_model, num_heads, dropout)(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

def build_transformer_model(input_shape, units=512, d_model=128, num_heads=4, dropout=0.3, num_layers=2):
    inputs = tf.keras.Input(shape=input_shape[1:])
    x = tf.keras.layers.Dense(d_model, kernel_regularizer=tf.keras.regularizers.l2(0.001))(inputs)
    x = stacked_transformer_encoder_decoder(units, d_model, num_heads, dropout, num_layers)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(1, activation='linear', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(), metrics=['mae'])
    return model

def evaluate_and_predict(model, X, y, scaler, raw_data):
    train_size = int(0.9 * len(X))
    X_test, y_test = X[train_size:], y[train_size:]
    loss, mae = model.evaluate(X_test, y_test)
    print(f'Test loss: {loss}, Test MAE: {mae}')

    # Use only the last sequence from the test set for next day prediction
    last_sequence = np.expand_dims(X_test[-1], axis=0)
    next_day_price_scaled = model.predict(last_sequence)
    next_day_price = scaler.inverse_transform(next_day_price_scaled).flatten()[0]
    print(f"Last close price: {raw_data[-1]}")
    print(f"Next predicted close price: {next_day_price}")

def main():
    file_path = 'ITC.NS.csv'
    sequence_length = 60

    raw_data = load_csv_data(file_path)
    raw_data = raw_data[:-1]

    if raw_data is None or sequence_length >= len(raw_data):
        print("Exiting due to data load failure or invalid sequence length.")
        return

    X, y, scaler = preprocess_time_series_data(raw_data, sequence_length)

    # Splitting the data into training and validation sets (70% training, 10% validation)
    train_size = int(0.7 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    model = build_transformer_model(X_train.shape)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20),
        ModelCheckpoint('best_model.h5', save_best_only=True),
        LearningRateScheduler(lambda epoch, lr: lr * 1.0 if epoch > 0 and epoch % 10 == 0 else lr)
    ]
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), callbacks=callbacks)

    best_model = tf.keras.models.load_model('best_model.h5')
    evaluate_and_predict(best_model, X, y, scaler, raw_data)

if __name__ == '__main__':
    main()
