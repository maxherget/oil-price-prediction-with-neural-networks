import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_data = pd.read_csv('../data/AMZN.csv')  #liest das Dataset
print(test_data)
test_data = test_data[['Date', 'Close']]
# print(test_data)
test_data['Date'] = pd.to_datetime(test_data['Date'])
# plt.plot(data['Date'], data['Close'])       #macht den Graphen
# print(test_data)

from copy import deepcopy as dc


def prepare_data_for_lstm(data_frame, n_steps):
    data_frame = dc(data_frame)

    data_frame.set_index('Date', inplace=True)

    for i in range(1, n_steps + 1):
        data_frame[f'Close(t-{i})'] = data_frame['Close'].shift(i)

    data_frame.dropna(inplace=True)

    return data_frame


lookback_range = 7  # Range wie viele vorhergehende Werte überprüft werden zum guessen des aktuellen Wertes
shifted_dataframe = prepare_data_for_lstm(test_data, lookback_range)
# print(shifted_dataframe)

shifted_dataframe_as_np = shifted_dataframe.to_numpy()
# print(shifted_dataframe_as_np)
# print(shifted_dataframe_as_np.shape)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
shifted_dataframe_as_np = scaler.fit_transform(
    shifted_dataframe_as_np)  # skaliert die dataset werte auf -1 - 0
# print(shifted_dataframe_as_np)

X = shifted_dataframe_as_np[:, 1:]  # Alle Daten aus der vergangenheit aus denen die Werte predicted werden
y = shifted_dataframe_as_np[:, 0]  # der Predictor, also die erste Zeile
# print(X.shape, y.shape)

X = dc(np.flip(X, axis=1))
# print(X)

split_index = int(len(X) * 0.95)  # 95% der Daten werden zum trainieren genutzt , restl. 5 zum testen
# print(split_index)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

# es wird eine weitere dimension hinzugefügt welche für das ltsm notwendig ist

X_train = X_train.reshape((-1, lookback_range, 1))
X_test = X_test.reshape((-1, lookback_range, 1))
X_train = X_train.reshape((-1, lookback_range, 1))
X_test = X_test.reshape((-1, lookback_range, 1))

X_train = torch.tensor(X_train).float()  # wrapping in pytorch-tensoren
y_train = torch.tensor(y_train).float()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()

# umwandeln in dataset-objekt

from torch.utils.data import Dataset


class CoustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


train_dataset = CoustomDataset(X_train, y_train)
test_dataset = CoustomDataset(X_test, y_test)

# wrapping dataset in dataloader
from torch.utils.data import DataLoader

batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for _, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    print(x_batch.shape, y_batch.shape)
    break


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


model = LSTM(1, 4, 1)
model.to(device)
print(model)

learning_rate = 0.001
num_epochs = 10
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_one_epoch():
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        y_batch = y_batch.unsqueeze(1) #fixt das problem mit unterschiedlichen dimensionen


        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 99:  # print every 100 batches
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, avg. Loss: {1:.3f}'.format(batch_index + 1,
                                                         avg_loss_across_batches))
            running_loss = 0.0
    print()


def validate_one_epoch():
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        y_batch = y_batch.unsqueeze(1)  # Gleiche Anpassung wie oben.

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

        avg_loss_across_batches = running_loss / len(test_loader)

    print('Valid. Loss: {0:.3f}'.format(avg_loss_across_batches)) #durchschnittliche Verlust auf dem Validierungsdatensatz
    print('###########################################')
    print()


for epoch in range(num_epochs):
    train_one_epoch()
    validate_one_epoch()


with torch.no_grad():
    predicted = model(X_train.to(device)).to('cpu').numpy()

# zurückkonvertieren der Trainingsdaten-Werte von -1 bis 1 Scale auf og. Scale
train_predictions = predicted.flatten()

reconverter_dataset = np.zeros((X_train.shape[0], lookback_range + 1))
reconverter_dataset[:, 0] = train_predictions
reconverter_dataset = scaler.inverse_transform(reconverter_dataset)

train_predictions = dc(reconverter_dataset[:, 0])
train_predictions

reconverter_dataset = np.zeros((X_train.shape[0], lookback_range + 1))
reconverter_dataset[:, 0] = y_train.flatten()
reconverter_dataset = scaler.inverse_transform(reconverter_dataset)

new_y_train = dc(reconverter_dataset[:, 0])
new_y_train


# Graph zu der predicted/actual trainings-Daten
plt.figure()  # Startet eine neue Figur
plt.plot(new_y_train, label='Actual Close', color='blue')
plt.plot(train_predictions, label='Predicted Close', color='red')
plt.title('Training Data: Actual vs Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()

# zurückkonvertieren der Trainingsdaten-Werte von -1 bis 1 Scale auf og. Scale
test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten()

reconverter_dataset = np.zeros((X_test.shape[0], lookback_range + 1))
reconverter_dataset[:, 0] = test_predictions
reconverter_dataset = scaler.inverse_transform(reconverter_dataset)

test_predictions = dc(reconverter_dataset[:, 0])

reconverter_dataset = np.zeros((X_test.shape[0], lookback_range + 1))
reconverter_dataset[:, 0] = y_test.flatten()
reconverter_dataset = scaler.inverse_transform(reconverter_dataset)

new_y_test = dc(reconverter_dataset[:, 0])
new_y_test

# Graph zu der predicted/actual test-Daten
plt.figure()
plt.plot(new_y_test, label='Actual Close', color='blue')
plt.plot(test_predictions, label='Predicted Close', color='red')
plt.title('Test Data: Actual vs Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()





















