import optuna
import os
import inspect
import sqlite3


def create_study():
    # Determine the name of the calling file without the file extension
    stack = inspect.stack()
    caller_file = os.path.splitext(os.path.basename(stack[1].filename))[0]

    storage = optuna.storages.RDBStorage(
        url='sqlite:///optuna_study.db',
        engine_kwargs={
            'connect_args': {'timeout': 10}
        }
    )
    study = optuna.create_study(study_name=caller_file, direction='minimize', storage=storage, load_if_exists=True)
    print(f"Study '{caller_file}' created or loaded successfully.")
    return study


def delete_study(study_name):
    storage = 'sqlite:///optuna_study.db'
    print(f'Deleting study: {study_name}')
    optuna.delete_study(study_name=study_name, storage=storage)
    print(f'Study {study_name} deleted successfully')


def find_best_trial():
    storage = 'sqlite:///optuna_study.db'
    study_summaries = optuna.get_all_study_summaries(storage=storage)
    best_trial = None
    best_study_name = None

    for study_summary in study_summaries:
        study = optuna.load_study(study_name=study_summary.study_name, storage=storage)
        trial = study.best_trial
        if best_trial is None or (trial.value is not None and trial.value < best_trial.value):
            best_trial = trial
            best_study_name = study_summary.study_name

    if best_trial:
        print(f'Best Trial ID: {best_trial.number}')
        print(f'Loss Value: {best_trial.value}')
        print(f'Best Model Name: {best_study_name}')
        print('Best Hyperparameters:')
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
        print(f'Model Name: {study_name}')
        print('Best Hyperparameters:')
        for param_name, param_value in best_trial.params.items():
            print(f'  {param_name}: {param_value}')
    else:
        print(f'No trials found for study: {study_name}')


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
        print(f'Study {study_name} deleted successfully')

    print("All studies have been deleted.")


def list_all_studies_with_details():
    storage = 'sqlite:///optuna_study.db'
    study_summaries = optuna.get_all_study_summaries(storage=storage)
    study_details = []

    for summary in study_summaries:
        study_name = summary.study_name
        study = optuna.load_study(study_name=study_name, storage=storage)
        trial_count = len(study.trials)
        best_trial = study.best_trial

        if best_trial:
            study_details.append((study_name, trial_count, best_trial.value, best_trial.number, best_trial.params))
        else:
            study_details.append((study_name, trial_count, None, None, None))

    # Sort studies by the lowest loss value
    study_details.sort(key=lambda x: (x[2] is not None, x[2]))

    for detail in study_details:
        study_name, trial_count, best_value, best_trial_id, params = detail
        print(f"Model Name: {study_name}")
        print(f"Number of Trials: {trial_count}")
        if best_value is not None:
            print(f"Best Trial ID: {best_trial_id}")
            print(f"Best Trial Value: {best_value}")
            print("Best Hyperparameters:")
            for param_name, param_value in params.items():
                print(f"  {param_name}: {param_value}")
        else:
            print("No trials found.")
        print("")


if __name__ == "__main__"
    print("\nHyperparameter Studies:")
    print("================================================================================")
    count_all_trials()

    print("================================================================================")
    print("Best Trial:\n")
    find_best_trial()

    print("================================================================================")
    print("Summary of all Models:\n")
    list_all_studies_with_details()
