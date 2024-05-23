import optuna
import os
import inspect
import sqlite3

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


def delete_study(study_name):
    storage = 'sqlite:///optuna_study.db'
    print(f'{study_name} wird aus der Datenbank gelöscht')
    optuna.delete_study(study_name=study_name, storage=storage)
    print(f'Studie {study_name} erfolgreich gelöscht')


def find_best_trial():
    storage = 'sqlite:///optuna_study.db'
    study_summaries = optuna.get_all_study_summaries(storage=storage)
    best_trial = None

    for study_summary in study_summaries:
        study = optuna.load_study(study_name=study_summary.study_name, storage=storage)
        trial = study.best_trial
        if best_trial is None or trial.value < best_trial.value:
            best_trial = trial

    if best_trial:
        print(f'Best Trial ID: {best_trial.number}')
        print(f'Loss Value: {best_trial.value}')
        print(f'Study Name: {best_trial.study.study_name}')
        print('Hyperparameters:')
        for param_name, param_value in best_trial.params.items():
            print(f'  {param_name}: {param_value}')
    else:
        print('No trials found.')


def find_best_trial_in_study(study_name):
    storage = 'sqlite:///optuna_study.db'

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        print(f'No study found with the name: {study_name}')
        return

    best_trial = study.best_trial

    if best_trial:
        print(f'Best Trial ID: {best_trial.number}')
        print(f'Loss Value: {best_trial.value}')
        print(f'Study Name: {study_name}')
        print('Hyperparameters:')
        for param_name, param_value in best_trial.params.items():
            print(f'  {param_name}: {param_value}')
    else:
        print(f'No trials found for study: {study_name}')

def get_study_summaries():
    storage = 'sqlite:///optuna_study.db'
    study_summaries = optuna.get_all_study_summaries(storage=storage)

    for summary in study_summaries:
        print(f"Study Name: {summary.study_name}")
        print(f"Direction: {summary.direction}")
        print(f"Best Trial Number: {summary.best_trial_number}")
        print(f"Best Trial Value: {summary.best_trial_value}")
        print("")


def get_all_trials_from_study(study_name):
    storage = 'sqlite:///optuna_study.db'

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        print(f'No study found with the name: {study_name}')
        return

    trials = study.trials

    if trials:
        for trial in trials:
            print(f'Trial ID: {trial.number}')
            print(f'Value: {trial.value}')
            print(f'Params: {trial.params}')
            print(f'State: {trial.state}')
            print('')
    else:
        print(f'No trials found for study: {study_name}')


def stop_running_study(study_name):
    storage = 'sqlite:///optuna_study.db'

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        study.stop()
        print(f'Study {study_name} has been stopped.')
    except KeyError:
        print(f'No study found with the name: {study_name}')

def count_all_trials():
    storage = 'sqlite:///optuna_study.db'
    study_summaries = optuna.get_all_study_summaries(storage=storage)
    total_trials = 0

    for study_summary in study_summaries:
        study = optuna.load_study(study_name=study_summary.study_name, storage=storage)
        total_trials += len(study.trials)

    print(f'Total number of trials across all studies: {total_trials}')
    return total_trials


def delete_all_studies():
    storage = 'sqlite:///optuna_study.db'
    study_summaries = optuna.get_all_study_summaries(storage=storage)

    for summary in study_summaries:
        study_name = summary.study_name
        optuna.delete_study(study_name=study_name, storage=storage)
        print(f'Study {study_name} erfolgreich gelöscht')

    print("All studies have been deleted.")

if __name__ == "__main__":
    #delete_study('')
    delete_all_studies()
    find_best_trial()
    # delete_study('')
    # find_best_trial_in_study('')

