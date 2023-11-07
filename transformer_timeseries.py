import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

# Fetch data
ticker = 'AAPL'
data = yf.download(ticker, start='2010-01-01', end='2023-01-01')

# Calculate percentage change
data['Return'] = data['Close'].pct_change()
data.dropna(inplace=True)

# Preprocess with MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Volume', 'Return']].values)

# Convert to PyTorch tensor
scaled_data = torch.tensor(scaled_data, dtype=torch.float32)

# Create sequences and labels
def create_sequences(data, sequence_length):
    sequences = []
    labels = []
    
    for i in range(len(data) - sequence_length):
        sequence = data[i:i + sequence_length]
        label = data[i + sequence_length][[0, 3]]  # Open and Close prices
        sequences.append(sequence)
        labels.append(label)
    
    return torch.stack(sequences), torch.stack(labels)

sequence_length = 10
sequences, labels = create_sequences(scaled_data, sequence_length)

# Split data into train and test sets
train_size = int(0.85 * len(sequences))
train_sequences, train_labels = sequences[:train_size], labels[:train_size]
test_sequences, test_labels = sequences[train_size:], labels[train_size:]

# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, sequence_length, num_layers, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(input_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(input_dim, num_heads, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(input_dim * sequence_length, 2)  # Open and Close prices
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask
        
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = output.flatten(start_dim=1)
        output = self.decoder(output)
        return output
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_dim = 6  # number of features
num_layers = 2
num_heads = 4
dim_feedforward = 2048
dropout = 0.1
batch_size = 64
learning_rate = 0.001
num_epochs = 100

# Load data
train_data = TensorDataset(train_sequences, train_labels)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Initialize model
model = TransformerModel(input_dim, sequence_length, num_layers, num_heads, dim_feedforward, dropout).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for i, (seqs, labels) in enumerate(train_loader):
        seqs = seqs.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(seqs)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Test the model
model.eval()
with torch.no_grad():
    test_sequences = test_sequences.to(device)
    predictions = model(test_sequences).cpu()
    predictions = scaler.inverse_transform(predictions)

# Print the last prediction
print(f"Predicted next day's open price: {predictions[-1][0]}")
print(f"Predicted next day's close price: {predictions[-1][1]}")
