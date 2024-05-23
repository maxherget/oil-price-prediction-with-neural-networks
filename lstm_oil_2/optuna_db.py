import optuna
import os
import inspect

def create_study():
    # Ermittelt den Dateinamen der aufrufenden Datei ohne die Dateierweiterung
    stack = inspect.stack()
    caller_file = os.path.splitext(os.path.basename(stack[1].filename))[0]

    storage = optuna.storages.RDBStorage(
        url='sqlite:///optuna_study.db',
        engine_kwargs={
            'connect_args': {'timeout': 10}
        }
    )
    study = optuna.create_study(study_name=caller_file, direction='minimize', storage=storage, load_if_exists=True)
    return study


