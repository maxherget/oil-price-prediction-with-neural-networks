import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

data = pd.read_csv('../data/AMZN.csv')
print(data)

data['Date'] = pd.to_datetime(data['Date'])

plt.plot(data['Date'], data['Close'])

from copy import deepcopy as dc


def prepare_data_for_lstm(data_frame, n_steps):
    data_frame = dc(data_frame)

    data_frame.set_index('Date', inplace=True)

    for i in range(1, n_steps + 1):
        data_frame[f'Close(t-{i})'] = data_frame['Close'].shift(i)

    data_frame.dropna(inplace=True)

    return data_frame


lookback_range = 7
shifted_dataframe = prepare_data_for_lstm(data, lookback_range)
print(shifted_dataframe)

shifted_dataframe_as_np = shifted_dataframe.to_numpy()


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1,1))
shifted_dataframe_as_np = scaler.fit_transform(shifted_dataframe_as_np)

print(shifted_dataframe_as_np)
