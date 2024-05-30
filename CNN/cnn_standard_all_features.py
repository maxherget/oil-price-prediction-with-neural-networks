import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import matplotlib.dates as mdates
from Hyperparameter_DB.optuna_db_controller import get_best_trial_from_study
from data.data_reader import get_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Daten laden
test_data = get_data()
test_data['date'] = pd.to_datetime(test_data['date'])
test_data = test_data.set_index('date')[['open', 'high', 'low', 'volume', 'close']]

# Skalierung der Daten
def min_max_scaling(data):
    min_vals = data.min()
    max_vals = data.max()
    scaled_data = (data - min_vals) / (max_vals - min_vals)
    return scaled_data, min_vals, max_vals

scaled_data, min_vals, max_vals = min_max_scaling(test_data)

# Daten für CNN vorbereiten
def prepare_data_for_cnn(data_frame, n_steps):
    data_frame = data_frame.copy()
    output = []
    for i in range(len(data_frame) - n_steps):
        output.append(data_frame.iloc[i:(i + n_steps)].values)
    return np.array(output)

lookback_range = 7
shifted_data = prepare_data_for_cnn(scaled_data, lookback_range)

# Daten in Tensoren umwandeln
def create_tensors(data):
    X = data[:, :-1]
    y = data[:, -1][:, -1]  # Schlusskurse als Ziel
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

# CNN Modell
class CNNModel(nn.Module):
    def __init__(self, input_size, output_size, conv1_out_channels, conv2_out_channels, fc1_units):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_size[0], out_channels=conv1_out_channels, kernel_size=2, stride=1)
        self.conv2 = nn.Conv1d(in_channels=conv1_out_channels, out_channels=conv2_out_channels, kernel_size=2, stride=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(conv2_out_channels * ((input_size[1] - 2) // 2 - 1), fc1_units)
        self.fc2 = nn.Linear(fc1_units, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Umwandeln der Eingabe zu [Batch_Size, Features, Sequence_Length]
        x = self.pool(self.relu(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

best_trial = get_best_trial_from_study("cnn_standard_all_features_optuna")
print("" + "=" * 100)

input_size = X.shape[1]  # Anzahl der Features
output_size = 1  # Wir sagen die Schlusskurse voraus

if best_trial is not None:
    print("Best parameters for model pulled from DB and used for run")
    best_params = best_trial.params
    conv1_out_channels = best_params['conv1_out_channels']
    conv2_out_channels = best_params['conv2_out_channels']
    fc1_units = best_params['fc1_units']
    batch_size = best_params['batch_size']
    learn_rate = best_params['learn_rate']
    epochs = best_params['epochs']
else:
    print("No Hyperparameter data in DB for this Model, running with manually set values")
    conv1_out_channels = 20
    conv2_out_channels = 20
    fc1_units = 50
    batch_size = 16
    learn_rate = 0.01
    epochs = 50
print("" + "=" * 100)


input_size = (5, lookback_range)
output_size = 1

train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=best_params['batch_size'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=best_params['batch_size'], shuffle=False)

model = CNNModel(input_size, output_size, best_params['conv1_out_channels'], best_params['conv2_out_channels'], best_params['fc1_units']).to(device)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=best_params['learn_rate'])

# Training des Modells mit den besten Hyperparametern
train_losses = []
val_losses = []

for epoch in range(best_params['epochs']):
    model.train()
    batch_train_losses = []
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch.unsqueeze(-1))
        loss.backward()
        optimizer.step()
        batch_train_losses.append(loss.item())
    train_losses.append(np.mean(batch_train_losses))

    model.eval()
    batch_val_losses = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
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
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch.unsqueeze(-1))
        test_losses.append(loss.item())
        predictions.extend(y_pred.cpu().numpy())
        actuals.extend(y_batch.cpu().numpy())

test_loss = np.mean(test_losses)
print(f'Test Loss: {test_loss}')

# Vorhersagen und tatsächliche Werte zurückskalieren
def inverse_scaling(scaled_values, min_val, max_val):
    return scaled_values * (max_val - min_val) + min_val

actuals = inverse_scaling(np.array(actuals).reshape(-1, 1), min_vals['close'], max_vals['close']).flatten()
predictions = inverse_scaling(np.array(predictions).reshape(-1, 1), min_vals['close'], max_vals['close']).flatten()

# Visualisierung
plt.figure(figsize=(14, 5))
ax = plt.gca()
# Zeitachse anpassen: Tage von den tatsächlichen Daten verwenden
time_range = test_data.index[lookback_range + train_size + val_size: lookback_range + train_size + val_size + len(actuals)]
plt.plot(time_range, actuals, label='Actual Prices')
plt.plot(time_range, predictions, label='Predicted Prices')
plt.title('Crude Oil Prices Prediction on Test Data')
plt.xlabel('Time (Years)')
plt.ylabel('Price (USD)')
plt.legend()

# Formatter und Locator für halbe Jahre verwenden
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

# Optional: Minor Locator für Monate
ax.xaxis.set_minor_locator(mdates.MonthLocator())

plt.show()
