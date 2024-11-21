Max Herget 580332, Sämi Lastal 584806

### This application makes it possible to automatically find the best neural netwerk model with the best parameters for the time series predictions of the inserted data set

#### Our focus was the LSTM, but we added a lot of comparison models like RNN, GRU, ...

-You can start and train models specifically by executing the corresponding ***model.py*** in the Python console

-You can start respective hyperparameter testing studies for the models,
if you run the corresponding model_***optuna.py*** of the model in the Python console. 
How many Parameter trials per run can be changed by *study.optimize(objective, **n_trials=?**)*
Or u can start sveral in succession with the ***run_studies_for_models*** method from [optuna_db_controller](Hyperparameter_DB/optuna_db_controller.py)

-You can get an overall overview of the models and their results by running the main of the [optuna_db_controller](Hyperparameter_DB/optuna_db_controller.py).
Further already implemented methods for handling the work results can also be put in its main for further information outputs


these are the params which every models got tested with for comparability of models, they can be changed in the ***create_Study()*** Methode im [optuna_db_controller](Hyperparameter_DB/optuna_db_controller.py)

all models except CNN:
```python
    'hidden_layer_size': 50,
    'num_layers': 2,
    'batch_size': 16,
    'learn_rate': 0.01,
    'epochs': 50
```
CNN Models:
```python
        'conv1_out_channels': 32,
        'conv2_out_channels': 32,
        'fc1_units': 50,
        'batch_size': 16,
        'learn_rate': 0.01,
        'epochs': 50
```
