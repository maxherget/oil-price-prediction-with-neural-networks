import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from optuna_db import create_study
import optuna

# Seeds für Reproduzierbarkeit setzen
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Daten laden
data = pd.read_csv('../data/Crude_Oil_data.csv')
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')[['open', 'high', 'low', 'volume', 'close']]

# Skalierung der Daten
def min_max_scaling(data):
    min_vals = data.min()
    max_vals = data.max()
    scaled_data = (data - min_vals) / (max_vals - min_vals)
    return scaled_data, min_vals, max_vals

scaled_data, min_vals, max_vals = min_max_scaling(data)

# Daten für LSTM vorbereiten
def prepare_data_for_lstm(data_frame, n_steps):
    output = data_frame.copy()
    n_features = data_frame.shape[1]
    for i in range(1, n_steps + 1):
        for col in data_frame.columns:
            output[f'{col}(t-{i})'] = data_frame[col].shift(i)
    output.dropna(inplace=True)
    return output

lookback_range = 7
shifted_data = prepare_data_for_lstm(scaled_data, lookback_range)

# Daten in Tensoren umwandeln
def create_tensors(data_frame):
    X = data_frame.drop(['close'], axis=1).values
    y = data_frame['close'].values
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)
    return X, y

X, y = create_tensors(shifted_data)
dataset = TensorDataset(X, y)

# Datensatz in Trainings-, Validierungs- und Testdatensatz aufteilen
train_size = int(0.7 * len(dataset))  # 70% für Training
val_size = int(0.2 * len(dataset))    # 20% für Validierung
test_size = len(dataset) - train_size - val_size  # 10% für Test

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

# LSTM Modell
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        batch_size = input_seq.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_layer_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_layer_size).to(device)
        lstm_out, _ = self.lstm(input_seq, (h0, c0))
        lstm_out = lstm_out[:, -1, :]
        predictions = self.linear(lstm_out)
        return predictions

# Optuna-Studie erstellen
def objective(trial):
    input_size = X.shape[1]  # Anzahl der Features
    output_size = 1  # Wir sagen die Schlusskurse voraus
    hidden_layer_size = trial.suggest_int('hidden_layer_size', 10, 100)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    batch_size = trial.suggest_int('batch_size', 16, 128)
    learn_rate = trial.suggest_float('learn_rate', 1e-5, 1e-1)
    epochs = trial.suggest_int('epochs', 10, 100)  # Hyperparameter für die Anzahl der Epochen

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMModel(input_size, hidden_layer_size, output_size, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learn_rate)

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            if X_batch.ndim != 3:
                X_batch = X_batch.view(-1, 1, input_size)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(-1))
            loss.backward()
            optimizer.step()

    model.eval()
    val_losses = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            if X_batch.ndim != 3:
                X_batch = X_batch.view(-1, 1, input_size)
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(-1))
            val_losses.append(loss.item())

    return np.mean(val_losses)

# Optuna-Studie starten
study = create_study()
study.optimize(objective, n_trials=1)

# Beste Ergebnisse anzeigen
print('\nBest trial:')
trial = study.best_trial

print('Value: ', trial.value)
print('Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')
print('')

# # Verwendung der besten Hyperparameter für das endgültige Training und die Bewertung
# best_params = trial.params
# hidden_layer_size = best_params['hidden_layer_size']
# num_layers = best_params['num_layers']
# batch_size = best_params['batch_size']
# learn_rate = best_params['learn_rate']
# epochs = best_params['epochs']
#
# input_size = X.shape[1]
# output_size = 1
#
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
# model = LSTMModel(input_size, hidden_layer_size, output_size, num_layers).to(device)
# criterion = nn.MSELoss()
# optimizer = Adam(model.parameters(), lr=learn_rate)
#
# # Training des Modells mit den besten Hyperparametern
# train_losses = []
# val_losses = []
#
# for epoch in range(epochs):
#     model.train()
#     batch_train_losses = []
#     for X_batch, y_batch in train_loader:
#         optimizer.zero_grad()
#         if X_batch.ndim != 3:
#             X_batch = X_batch.view(-1, 1, input_size)
#         y_pred = model(X_batch)
#         loss = criterion(y_pred, y_batch.unsqueeze(-1))
#         loss.backward()
#         optimizer.step()
#         batch_train_losses.append(loss.item())
#     train_losses.append(np.mean(batch_train_losses))
#
#     model.eval()
#     batch_val_losses = []
#     with torch.no_grad():
#         for X_batch, y_batch in val_loader:
#             if X_batch.ndim != 3:
#                 X_batch = X_batch.view(-1, 1, input_size)
#             y_pred = model(X_batch)
#             loss = criterion(y_pred, y_batch.unsqueeze(-1))
#             batch_val_losses.append(loss.item())
#     val_losses.append(np.mean(batch_val_losses))
#
#     print(f'Epoch {epoch + 1}, Train Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}')
#
# # Lernkurven visualisieren um Overfitting sichtbarer zu machen
# plt.figure(figsize=(10, 6))
# plt.plot(train_losses, label='Train Loss')
# plt.plot(val_losses, label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.title('Train and Validation Loss over Epochs')
# plt.show()
#
# # Modell evaluieren
# model.eval()
# test_losses = []
# predictions = []
# actuals = []
# with torch.no_grad():
#     for X_batch, y_batch in test_loader:
#         if X_batch.ndim != 3:
#             X_batch = X_batch.view(-1, 1, input_size)
#         y_pred = model(X_batch)
#         loss = criterion(y_pred, y_batch.unsqueeze(-1))
#         test_losses.append(loss.item())
#         predictions.extend(y_pred.cpu().numpy())
#         actuals.extend(y_batch.cpu().numpy())
#
# test_loss = np.mean(test_losses)
# print(f'Test Loss: {test_loss}')
#
# # Vorhersagen und tatsächliche Werte zurückskalieren
# def inverse_scaling(scaled_values, min_val, max_val):
#     return scaled_values * (max_val - min_val) + min_val
#
# predictions = inverse_scaling(np.array(predictions).reshape(-1, 1), min_vals['close'], max_vals['close']).flatten()
# actuals = inverse_scaling(np.array(actuals).reshape(-1, 1), min_vals['close'], max_vals['close']).flatten()
#
# # Visualisierung der Vorhersagen und der tatsächlichen Werte
# plt.figure(figsize=(14, 5))
# dates = data.index[lookback_range + train_size:lookback_range + train_size + len(actuals)]
# plt.plot(dates, actuals, label='Actual Prices')
# plt.plot(dates, predictions, label='Predicted Prices')
# plt.title('Crude Oil Prices Prediction on Test Data')
# plt.xlabel('Time (Days)')
# plt.ylabel('Price (USD)')
# plt.legend()
# plt.show()
