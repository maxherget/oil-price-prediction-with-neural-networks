import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Laden des Datensatzes
test_data = pd.read_csv('../data/AMZN.csv')  # Ändern Sie den Pfad zu Ihrem Datensatz
print(test_data)

# Vorbereiten der Daten für das RNN
test_data = test_data[['Date', 'Close']]
test_data['Date'] = pd.to_datetime(test_data['Date'])

lookback_range = 7  # Anzahl der zu berücksichtigenden vergangenen Werte
shifted_dataframe = prepare_data_for_lstm(test_data, lookback_range)

# Normalisieren der Daten
shifted_dataframe_as_np = shifted_dataframe.to_numpy()
scaler = MinMaxScaler(feature_range=(-1, 1))
shifted_dataframe_as_np = scaler.fit_transform(shifted_dataframe_as_np)

# Trennen von X (Eingabedaten) und y (Zielwerte)
X = shifted_dataframe_as_np[:, 1:]  # Alle Daten aus der Vergangenheit
y = shifted_dataframe_as_np[:, 0]  # Der zu prognostizierende Wert

# Umkehren der Reihenfolge der Eingabedaten
X = dc(np.flip(X, axis=1))

# Aufteilen der Daten in Trainings- und Testdatensätze
split_index = int(len(X) * 0.95)
X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

# Umformen der Daten in die richtige Dimension für das RNN
X_train = X_train.reshape((-1, lookback_range, 1))
X_test = X_test.reshape((-1, lookback_range, 1))

# Konvertieren in PyTorch-Tensoren
X_train = torch.tensor(X_train).float().to(device)
y_train = torch.tensor(y_train).float().to(device)
X_test = torch.tensor(X_test).float().to(device)
y_test = torch.tensor(y_test).float().to(device)

# Definieren des Datensatzes
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)

# Laden der Datensätze in den DataLoader
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Definieren des RNN-Modells
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.rnn(x)  # Ignorieren Sie den verborgenen Zustand
        output = self.fc(output[:, -1, :])  # Nehmen Sie die letzte Ausgabe der RNN
        return output

model = RNN(1, 4, 2).to(device)  # Anpassen von input_size, hidden_size und num_layers
print(model)

# Definieren der Trainings- und Validierungsfunktionen
learning_rate = 0.001
num_epochs = 10
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
def train_one_epoch():
    model.train()
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        y_batch = y_batch.unsqueeze(1)  # Anpassung der Dimensionen

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 99:  # Ausgabe alle 100 Batches
            avg_loss_across_batches = running_loss / 100
            print(f'Batch {batch_index + 1}, avg. Loss: {avg_loss_across_batches:.3f}')
            running_loss = 0.0
    print()

def validate_one_epoch():
    model.eval()
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        y_batch = y_batch.unsqueeze(1)  # Gleiche Anpassung wie oben

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)
    print(f'Valid. Loss: {avg_loss_across_batches:.3f}')  # Ausgabe des Validierungsverlusts
    print('###########################################')
    print()

# Training des Modells
for epoch in range(num_epochs):
    train_one_epoch()
    validate_one_epoch()

# Vorhersagen auf dem Trainingsdatensatz
with torch.no_grad():
    predicted = model(X_train.to(device)).to('cpu').numpy()

# Rückkonvertierung der Trainingsdaten von -1 bis 1 auf den ursprünglichen Maßstab
train_predictions = predicted.flatten()
reconverter_dataset = np.zeros((X_train.shape[0], lookback_range + 1))
reconverter_dataset[:, 0] = train_predictions
reconverter_dataset = scaler.inverse_transform(reconverter_dataset)

train_predictions = dc(reconverter_dataset[:, 0])
new_y_train = dc(reconverter_dataset[:, 1])

# Visualisierung der Vorhersagen vs. tatsächliche Werte im Trainingsdatensatz
plt.figure()
plt.plot(new_y_train, label='Actual Close', color='blue')
plt.plot(train_predictions, label='Predicted Close', color='red')
plt.title('Trainingsdaten: Tatsächliche vs. prognostizierte Schlusskurse')
plt.xlabel('Tag')
plt.ylabel('Schlusskurs')
plt.legend()
plt.show()

# Vorhersagen auf dem Testdatensatz
with torch.no_grad():
    test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten()

# Rückkonvertierung der Testdaten von -1 bis 1 auf den ursprünglichen Maßstab
reconverter_dataset = np.zeros((X_test.shape[0], lookback_range + 1))
reconverter_dataset[:, 0] = test_predictions
reconverter_dataset = scaler.inverse_transform(reconverter_dataset)

test_predictions = dc(reconverter_dataset[:, 0])
new_y_test = dc(reconverter_dataset[:, 1])

# Visualisierung der Vorhersagen vs. tatsächliche Werte im Testdatensatz
plt.figure()
plt.plot(new_y_test, label='Actual Close', color='blue')
plt.plot(test_predictions, label='Predicted Close', color='red')
plt.title('Testdaten: Tatsächliche vs. prognostizierte Schlusskurse')
plt.xlabel('Tag')
plt.ylabel('Schlusskurs')
plt.legend()
plt.show()
