import optuna
import os
import sys
import inspect
import subprocess

from optuna.storages import RDBStorage

db_relative_path = os.path.join(os.path.dirname(__file__), 'optuna_study.db')
db_absolute_path = os.path.abspath(db_relative_path)

sqlite_url = f'sqlite:///{db_absolute_path}'

def run_studies_for_models(scripts):
    python_executable = sys.executable  # Holt sich den aktuellen Python-Interpreter
    for script in scripts:
        script_path = os.path.abspath(script)  # Absoluten Pfad zum Skript verwenden

        # Befehl zum Ausführen des Skripts
        result = subprocess.run([python_executable, script_path], capture_output=True, text=True)

        # Ausgabe des Skripts anzeigen
        print(f"\nOutput for {script_path}:\n")
        if result.stderr:
            print(result.stderr)
        print(result.stdout)
        print("" + "=" * 100)

# def create_study():
#     # Determine the name of the calling file without the file extension
#     stack = inspect.stack()
#     caller_file = os.path.splitext(os.path.basename(stack[1].filename))[0]
#
#     storage = optuna.storages.RDBStorage(
#         url='sqlite:///optuna_study.db',
#         engine_kwargs={
#             'connect_args': {'timeout': 10}
#         }
#     )
#     study = optuna.create_study(study_name=caller_file, direction='minimize', storage=storage, load_if_exists=True)
#     print(f"Study '{caller_file}' created or loaded successfully.")
#     return study


def create_study():
    stack = inspect.stack()
    caller_file = os.path.splitext(os.path.basename(stack[1].filename))[0]

    storage = RDBStorage(
        url='sqlite:///optuna_study.db',
        engine_kwargs={
            'connect_args': {'timeout': 10}
        }
    )

    best_trials = get_best_trials_from_study(caller_file)
    if not best_trials.empty:
        best_params = best_trials[
            ['params_hidden_layer_size', 'params_num_layers', 'params_batch_size', 'params_learn_rate']].to_dict(
            'records')
    else:
        best_params = []

    # Initialize TPESampler
    sampler = optuna.samplers.TPESampler(seed=0, n_startup_trials=10, multivariate=True, warn_independent_sampling= False)
    # constant_liar = True -> attribe for sampler for running many trials parallel.

    study = optuna.create_study(
        study_name=caller_file,
        direction='minimize',
        storage=storage,
        load_if_exists=True,
        sampler=sampler
    )

    for params in best_params:
        study.enqueue_trial(params)

    print(f"Study '{caller_file}' created or loaded successfully.")
    return study


def delete_study(study_name):
    storage = 'sqlite:///optuna_study.db'
    print(f'Deleting study: {study_name}')
    optuna.delete_study(study_name=study_name, storage=storage)
    print(f'Study {study_name} deleted successfully')


def transfer_trials(source_study_name, target_study_name):
    storage = RDBStorage(url='sqlite:///optuna_study.db')

    source_study = optuna.load_study(study_name=source_study_name, storage=storage)
    target_study = optuna.load_study(study_name=target_study_name, storage=storage)

    # Transfer each trial from the source study to the target study
    for trial in source_study.trials:
        target_study.add_trial(trial)

    print(f"Transferred {len(source_study.trials)} trials from study {source_study_name} to study {target_study_name}")


def get_best_trial():
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
    return best_trial




def get_best_trial_from_study(study_name):
    storage = RDBStorage(
        url=sqlite_url,
        engine_kwargs={
            'connect_args': {'timeout': 10}
        }
    )

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
    return best_trial


def get_best_trials_from_study(study_name, n_best=5):
    storage = RDBStorage(url='sqlite:///optuna_study.db')
    study = optuna.load_study(study_name=study_name, storage=storage)
    trials = study.trials_dataframe()
    best_trials = trials.sort_values('value').head(n_best)
    return best_trials


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
    return trials


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

def get_trial_with_highest_loss_from_study(study_name):
    storage = RDBStorage(
        url=sqlite_url,
        engine_kwargs={
            'connect_args': {'timeout': 10}
        }
    )

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        print(f'No study found with the name: {study_name}')
        return

    # Find the trial with the highest loss
    trials = study.trials
    if not trials:
        print(f'No trials found for study: {study_name}')
        return

    highest_loss_trial = max(trials, key=lambda t: t.value if t.value is not None else float('-inf'))

    print(f'Trial ID with Highest Loss: {highest_loss_trial.number}')
    print(f'Loss Value: {highest_loss_trial.value}')
    print(f'Model Name: {study_name}')
    print('Hyperparameters:')
    for param_name, param_value in highest_loss_trial.params.items():
        print(f'  {param_name}: {param_value}')

    return highest_loss_trial


def get_trial_with_highest_loss_overall():
    storage = 'sqlite:///optuna_study.db'
    study_summaries = optuna.get_all_study_summaries(storage=storage)
    worst_trial = None
    worst_study_name = None

    for study_summary in study_summaries:
        study = optuna.load_study(study_name=study_summary.study_name, storage=storage)
        for trial in study.trials:
            if worst_trial is None or (trial.value is not None and trial.value > worst_trial.value):
                worst_trial = trial
                worst_study_name = study_summary.study_name

    if worst_trial:
        print(f'Worst Trial ID: {worst_trial.number}')
        print(f'Loss Value: {worst_trial.value}')
        print(f'Worst Model Name: {worst_study_name}')
        print('Worst Hyperparameters:')
        for param_name, param_value in worst_trial.params.items():
            print(f'  {param_name}: {param_value}')
    else:
        print('No trials found.')

    return worst_trial


def count_studies():
    storage = RDBStorage(url='sqlite:///optuna_study.db')
    study_summaries = optuna.get_all_study_summaries(storage=storage)
    num_studies = len(study_summaries)
    print(f"Total number of studies for models: {num_studies}")
    return num_studies


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
    return study_details


if __name__ == "__main__":
    models_to_run = [
        # 'lstm_antiOverfit_optuna.py',
        # 'lstm_antiOverfit_all_features_optuna.py',
        # 'lstm_antiOverfit_temp_attention_optuna.py',
        # 'lstm_standard_all_features_optuna.py',
        # 'lstm_standard_optuna.py',
        # 'lstm_temp_attention_all_features_optuna.py',
        # 'lstm_temp_attention_optuna.py'
    ]
    run_studies_for_models(models_to_run)


    print("\nOverall statistics:")
    print("" + "=" * 100)
    count_studies()
    count_all_trials()

    print("" + "=" * 100)
    print("Best Trial overall:\n")
    get_best_trial()

    print("" + "=" * 100)
    print("Summary of all Models:\n")
    list_all_studies_with_details()
