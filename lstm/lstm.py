import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch import device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_data = pd.read_csv('../data/AMZN.csv')  #liest das Dataset
# print(test_data)
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

for _, batch in enumerate(train_loader):
    x_batch, y_batch = batch[0].to(device), batch[1].to(device)
    print(x_batch.shape, y_batch.shape)
    break

