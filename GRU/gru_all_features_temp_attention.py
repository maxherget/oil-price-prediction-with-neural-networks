import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
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

# Daten für RNN vorbereiten
def prepare_data_for_rnn(data_frame, n_steps):
    output = data_frame.copy()
    n_features = data_frame.shape[1]
    for i in range(1, n_steps + 1):
        for col in data_frame.columns:
            output[f'{col}(t-{i})'] = data_frame[col].shift(i)
    output.dropna(inplace=True)
    return output

lookback_range = 7
shifted_data = prepare_data_for_rnn(scaled_data, lookback_range)

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

# Attention Layer
class AttentionLayer(nn.Module):
    def __init__(self, hidden_layer_size):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Parameter(torch.Tensor(hidden_layer_size, 1))
        nn.init.xavier_uniform_(self.attention_weights)

    def forward(self, hidden_states):
        # hidden_states: [batch_size, seq_len, hidden_layer_size]
        scores = torch.matmul(hidden_states, self.attention_weights).squeeze(-1)  # [batch_size, seq_len]
        attention_weights = torch.softmax(scores, dim=1)  # [batch_size, seq_len]
        context_vector = torch.matmul(attention_weights.unsqueeze(1), hidden_states).squeeze(1)  # [batch_size, hidden_layer_size]
        return context_vector, attention_weights

# GRU Modell mit Attention
class GRUModelWithAttention(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size, num_layers):
        super(GRUModelWithAttention, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.attention = AttentionLayer(hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        gru_out, _ = self.gru(input_seq)
        context_vector, attention_weights = self.attention(gru_out)
        predictions = self.linear(context_vector)
        return predictions, attention_weights

# Training des Modells mit den besten Hyperparametern
best_trial = get_best_trial_from_study("gru_standard_all_features_optuna")
print("" + "=" * 100)

input_size = X.shape[1]  # Anzahl der Features
output_size = 1  # Wir sagen die Schlusskurse voraus

if best_trial is not None:
    print("Best parameters for model pulled from DB and used for run")
    best_params = best_trial.params
    hidden_layer_size = best_params['hidden_layer_size']
    num_layers = best_params['num_layers']
    batch_size = best_params['batch_size']
    learn_rate = best_params['learn_rate']
    epochs = best_params['epochs']
else:
    print("No Hyperparameter data in DB for this Model, running with manually set values")
    hidden_layer_size = 50
    num_layers = 2
    batch_size = 16
    learn_rate = 0.01
    epochs = 50
print("" + "=" * 100)


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = GRUModelWithAttention(input_size, hidden_layer_size, output_size, num_layers).to(device)
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=learn_rate)

train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    batch_train_losses = []
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        if X_batch.ndim != 3:
            X_batch = X_batch.view(-1, 1, input_size)
        y_pred, _ = model(X_batch)
        loss = criterion(y_pred, y_batch.unsqueeze(-1))
        loss.backward()
        optimizer.step()
        batch_train_losses.append(loss.item())
    train_losses.append(np.mean(batch_train_losses))

    model.eval()
    batch_val_losses = []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            if X_batch.ndim != 3:
                X_batch = X_batch.view(-1, 1, input_size)
            y_pred, _ = model(X_batch)
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
        if X_batch.ndim != 3:
            X_batch = X_batch.view(-1, 1, input_size)
        y_pred, _ = model(X_batch)
        loss = criterion(y_pred, y_batch.unsqueeze(-1))
        test_losses.append(loss.item())
        predictions.extend(y_pred.cpu().numpy())
        actuals.extend(y_batch.cpu().numpy())

test_loss = np.mean(test_losses)
print(f'Test Loss: {test_loss}')

# Vorhersagen und tatsächliche Werte zurückskalieren
def inverse_scaling(scaled_values, min_val, max_val):
    return scaled_values * (max_val - min_val) + min_val

predictions = inverse_scaling(np.array(predictions).reshape(-1, 1), min_vals['close'], max_vals['close']).flatten()
actuals = inverse_scaling(np.array(actuals).reshape(-1, 1), min_vals['close'], max_vals['close']).flatten()

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
