import optuna
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
import torch.nn as nn


def objective(trial, test_data, prepare_data_for_lstm, create_tensors, model_config, device):
    lookback_range = trial.suggest_int('lookback', 5, 50)
    hidden_layer_size = trial.suggest_int('hidden_layer_size', 10, 100)
    batch_size = trial.suggest_int('batch_size', 16, 64)
    epochs = trial.suggest_int('epochs', 50, 200)

    shifted_dataframe = prepare_data_for_lstm(test_data, lookback_range)
    X, y = create_tensors(shifted_dataframe)
    dataset = TensorDataset(X, y)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model_class = model_config['model_class']
    output_size = model_config['output_size']
    learn_rate = model_config['learn_rate']
    loss_function = model_config['loss_function']

    # input_size wird hier nicht direkt genutzt, weil es im Model Konstruktor verwendet wird.
    # Es wird angenommen, dass model_class den richtigen input_size intern verwendet.
    model = model_class(lookback_range, hidden_layer_size, output_size).to(device)

    criterion = loss_function
    optimizer = Adam(model.parameters(), lr=learn_rate)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch.unsqueeze(-1))
            loss = criterion(y_pred, y_batch.unsqueeze(-1))
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                y_pred = model(X_batch.unsqueeze(-1))
                loss = criterion(y_pred, y_batch.unsqueeze(-1))
                val_losses.append(loss.item())

        # Die Variable train_loss wird jetzt verwendet
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

    return val_loss


def run_study(test_data, prepare_data_for_lstm, create_tensors, model_config):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, test_data, prepare_data_for_lstm, create_tensors, model_config,
                                           torch.device("cuda" if torch.cuda.is_available() else "cpu")), n_trials=1)
    print("Beste Hyperparameter: ", study.best_params)
    return study.best_params

