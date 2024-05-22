import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from copy import deepcopy as dc

# Seeds für Reproduzierbarkeit setzen
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Daten laden
test_data = pd.read_csv('../data/Crude_Oil_data.csv')
test_data = test_data[['date', 'open', 'high', 'low', 'close', 'volume']]
test_data['date'] = pd.to_datetime(test_data['date'])

# Manuelle Skalierung der Daten
def min_max_scaling(data):
    min_val = np.min(data)
    max_val = np.max(data)
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data, min_val, max_val

# Manuelle Umkehrung der Skalierung
def inverse_min_max_scaling(scaled_data, min_val, max_val):
    return scaled_data * (max_val - min_val) + min_val

scaled_data = {}
for column in ['open', 'high', 'low', 'close', 'volume']:
    test_data[column], scaled_data[f'{column}_min'], scaled_data[f'{column}_max'] = min_max_scaling(test_data[column])

def prepare_data_for_rnn(data_frame, n_steps):
    data_frame = dc(data_frame)
    data_frame.set_index('date', inplace=True)
    features = ['open', 'high', 'low', 'volume']
    for feature in features:
        for i in range(1, n_steps + 1):
            data_frame[f'{feature}(t-{i})'] = data_frame[feature].shift(i)
    data_frame.dropna(inplace=True)
    return data_frame

lookback_range = 7
shifted_dataframe = prepare_data_for_rnn(test_data, lookback_range)

# Daten in Tensoren umwandeln
def create_tensors(data_frame, target_col='close'):
    features = data_frame.drop(target_col, axis=1).values
    target = data_frame[target_col].values
    features = torch.tensor(features, dtype=torch.float32).to(device)
    target = torch.tensor(target, dtype=torch.float32).to(device)
    return features, target

X, y = create_tensors(shifted_dataframe)
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# RNN Modell definieren
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.rnn = nn.RNN(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        batch_size = input_seq.size(0)  # Batchgröße extrahieren
        seq_len = input_seq.size(1)  # Sequenzlänge extrahieren
        rnn_out, _ = self.rnn(input_seq)
        predictions = self.linear(rnn_out[:, -1, :])  # Letzte Ausgabe der RNN-Schicht verwenden
        return predictions

input_size = shifted_dataframe.shape[1] - 1  # Anzahl der Merkmale abzüglich der Zielgröße
hidden_layer_size = 50
output_size = 1

model = RNNModel(input_size, hidden_layer_size, output_size).to(device)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training des Modells
epochs = 100

for epoch in range(epochs):
    model.train()
    train_losses = []
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        X_batch = X_batch.view(X_batch.size(0), 1, -1)  # Größe (Batch, Sequenzlänge, Inputgröße) herstellen
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch.unsqueeze(-1))
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    val_losses = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.view(X_batch.size(0), 1, -1)  # Größe (Batch, Sequenzlänge, Inputgröße) herstellen
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(-1))
            val_losses.append(loss.item())

    train_loss = np.mean(train_losses)
    val_loss = np.mean(val_losses)

    print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss}')

# Lernkurven visualisieren um Overfitting sichtbarer zu machen
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Train and Validation Loss over Epochs')
plt.show()

# Modell evaluieren
model.eval()
test_losses = []
predictions = []
actuals = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.view(X_batch.size(0), 1, -1)  # Größe (Batch, Sequenzlänge, Inputgröße) herstellen
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch.unsqueeze(-1))
        test_losses.append(loss.item())
        predictions.extend(y_pred.cpu().numpy())
        actuals.extend(y_batch.cpu().numpy())

test_loss = np.mean(test_losses)
print(f'Test Loss: {test_loss}')

# Vorhersagen und tatsächliche Werte skalieren
actuals = inverse_min_max_scaling(np.array(actuals).reshape(-1, 1), scaled_data['close_min'], scaled_data['close_max']).flatten()
predictions = inverse_min_max_scaling(np.array(predictions).reshape(-1, 1), scaled_data['close_min'], scaled_data['close_max']).flatten()

# Visualisierung
plt.figure(figsize=(14, 5))
# Zeitachse anpassen: Tage von den tatsächlichen Daten verwenden
time_range = test_data.index[lookback_range + train_size : lookback_range + train_size + len(actuals)]
plt.plot(time_range, actuals, label='Actual Prices')
plt.plot(time_range, predictions, label='Predicted Prices')
plt.title('Crude Oil Prices Prediction on Test Data')
plt.xlabel('Time (Days)')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
