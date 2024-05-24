import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from copy import deepcopy as dc
import optuna

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
    shifted_columns = []
    for feature in features:
        for i in range(1, n_steps + 1):
            shifted_columns.append(data_frame[feature].shift(i).rename(f'{feature}(t-{i})'))
    shifted_dataframe = pd.concat([data_frame] + shifted_columns, axis=1)
    shifted_dataframe.dropna(inplace=True)
    return shifted_dataframe


def create_tensors(data_frame, target_col='close'):
    features = data_frame.drop(target_col, axis=1).values
    target = data_frame[target_col].values
    features = torch.tensor(features, dtype=torch.float32).to(device)
    target = torch.tensor(target, dtype=torch.float32).to(device)
    return features, target


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(RNNModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.rnn = nn.RNN(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        rnn_out, _ = self.rnn(input_seq)
        predictions = self.linear(rnn_out[:, -1, :])
        return predictions


# Optuna-Ziel-Funktion
def objective(trial):
    lookback_range = trial.suggest_int('lookback', 5, 50)
    hidden_layer_size = trial.suggest_int('hidden_layer_size', 10, 100)
    batch_size = trial.suggest_int('batch_size', 16, 64)
    epochs = trial.suggest_int('epochs', 50, 200)

    shifted_dataframe = prepare_data_for_rnn(test_data, lookback_range)
    X, y = create_tensors(shifted_dataframe)
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = RNNModel(shifted_dataframe.shape[1] - 1, hidden_layer_size, 1).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch.view(X_batch.size(0), 1, -1))
            loss = criterion(y_pred, y_batch.unsqueeze(-1))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_pred = model(X_batch.view(X_batch.size(0), 1, -1))
                loss = criterion(y_pred, y_batch.unsqueeze(-1))
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

    return val_loss


# Optuna-Studie erstellen und starten
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50)

print("Beste Hyperparameter: ", study.best_params)

# Mit besten Hyperparametern trainieren und evaluieren
best_lookback = study.best_params['lookback']
best_hidden_layer_size = study.best_params['hidden_layer_size']
best_batch_size = study.best_params['batch_size']
best_epochs = study.best_params['epochs']

shifted_dataframe = prepare_data_for_rnn(test_data, best_lookback)
X, y = create_tensors(shifted_dataframe)
dataset = TensorDataset(X, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=best_batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=best_batch_size, shuffle=False)

model = RNNModel(shifted_dataframe.shape[1] - 1, best_hidden_layer_size, 1).to(device)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)

train_losses = []
val_losses = []

for epoch in range(best_epochs):
    model.train()
    batch_train_losses = []
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch.view(X_batch.size(0), 1, -1))
        loss = criterion(y_pred, y_batch.unsqueeze(-1))
        loss.backward()
        optimizer.step()
        batch_train_losses.append(loss.item())
    train_losses.append(np.mean(batch_train_losses))

    model.eval()
    batch_val_losses = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_pred = model(X_batch.view(X_batch.size(0), 1, -1))
            loss = criterion(y_pred, y_batch.unsqueeze(-1))
            batch_val_losses.append(loss.item())
    val_losses.append(np.mean(batch_val_losses))

    print(f'Epoch {epoch + 1}, Train Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}')

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
        y_pred = model(X_batch.view(X_batch.size(0), 1, -1))
        loss = criterion(y_pred, y_batch.unsqueeze(-1))
        test_losses.append(loss.item())
        predictions.extend(y_pred.cpu().numpy())
        actuals.extend(y_batch.cpu().numpy())

test_loss = np.mean(test_losses)
print(f'Test Loss: {test_loss}')

# Vorhersagen und tatsächliche Werte skalieren
actuals = inverse_min_max_scaling(np.array(actuals).reshape(-1, 1), scaled_data['close_min'],
                                  scaled_data['close_max']).flatten()
predictions = inverse_min_max_scaling(np.array(predictions).reshape(-1, 1), scaled_data['close_min'],
                                      scaled_data['close_max']).flatten()

# Visualisierung
plt.figure(figsize=(14, 5))
time_range = test_data.index[best_lookback + train_size: best_lookback + train_size + len(actuals)]
plt.plot(time_range, actuals, label='Actual Prices')
plt.plot(time_range, predictions, label='Predicted Prices')
plt.title('Crude Oil Prices Prediction on Test Data')
plt.xlabel('Time (Days)')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
