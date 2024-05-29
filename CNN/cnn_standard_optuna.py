import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from Hyperparameter_DB.optuna_db_controller import create_study

# Seeds für Reproduzierbarkeit setzen
np.random.seed(0)
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Daten laden
data = pd.read_csv('../data/Crude_Oil_data.csv')
data['date'] = pd.to_datetime(data['date'])
data = data.set_index('date')[['close']]  # Nur 'close' Wert behalten

# Skalierung der Daten
def min_max_scaling(data):
    min_vals = data.min()
    max_vals = data.max()
    scaled_data = (data - min_vals) / (max_vals - min_vals)
    return scaled_data, min_vals, max_vals

scaled_data, min_vals, max_vals = min_max_scaling(data)

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

# Optuna-Studie erstellen
def objective(trial):
    input_size = (1, lookback_range)  # Eingabegröße anpassen, da nur ein Feature ('close') betrachtet wird
    output_size = 1  # Wir sagen die Schlusskurse voraus
    conv1_out_channels = trial.suggest_int('conv1_out_channels', 10, 50)
    conv2_out_channels = trial.suggest_int('conv2_out_channels', 10, 50)
    fc1_units = trial.suggest_int('fc1_units', 10, 100)
    batch_size = trial.suggest_int('batch_size', 16, 128)
    learn_rate = trial.suggest_float('learn_rate', 1e-3, 1e-1)
    epochs = trial.suggest_int('epochs', 10, 100)  # Hyperparameter für die Anzahl der Epochen

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CNNModel(input_size, output_size, conv1_out_channels, conv2_out_channels, fc1_units).to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=learn_rate)

    for epoch in range(epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(-1))
            loss.backward()
            optimizer.step()

    model.eval()
    val_losses = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch.unsqueeze(-1))
            val_losses.append(loss.item())

    return np.mean(val_losses)

# Optuna-Studie starten
study = create_study()
study.optimize(objective, n_trials=10)

# Beste Ergebnisse anzeigen
print('\nBest trial:')
trial = study.best_trial

print('Value: ', trial.value)
print('Params: ')
for key, value in trial.params.items():
    print(f'    {key}: {value}')
print('')
