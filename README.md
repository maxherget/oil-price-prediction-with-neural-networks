### This application makes it possible to automatically find the best model with the best parameters for the time series predictions of the inserted data set

#### Our focus was the LSTM, but we added a lot of comparison models

-You can start and train models specifically by executing the corresponding ***model.py*** in the Python console

-You can start respective hyperparameter testing studies for the models,
if you run the corresponding model_***optuna.py*** of the model in the Python console. 
How many Parameter trials per run can be changed by *study.optimize(objective, **n_trials=?**)*

-You can get an overall overview of the models and their results by running the main of the [optuna_db_controller](Hyperparameter_DB/optuna_db_controller.py).
Further already implemented methods for handling the work results can also be put in its main for further information outputs


### todo:
- Ergebnisse für Präsentation vorbereiten
