import optuna
import os
import sys
import inspect
import subprocess

from optuna.storages import RDBStorage

db_relative_path = os.path.join(os.path.dirname(__file__), 'optuna_study.db')
db_absolute_path = os.path.abspath(db_relative_path)

sqlite_url = f'sqlite:///{db_absolute_path}'


def find_project_root_directory():
    current_directory = os.path.dirname(os.path.abspath(__file__))

    while current_directory != os.path.dirname(current_directory):
        if os.path.exists(os.path.join(current_directory, 'README.md')):
            return current_directory
        current_directory = os.path.dirname(current_directory)

    raise Exception("root directory could not be found.")


def find_script_path(script_name, root_directory, script_extension='.py'):
    for root, _, files in os.walk(root_directory):
        for file in files:
            if file == f"{script_name}{script_extension}":
                return os.path.join(root, file)
    return None


def run_studies_for_models(scripts):
    python_executable = sys.executable
    project_root_directory = find_project_root_directory()

    for script in scripts:
        script_path = find_script_path(script, project_root_directory)
        if script_path:
            script_path = os.path.abspath(script_path)
            # Befehl zum Ausführen des Skripts
            result = subprocess.run([python_executable, script_path], capture_output=True, text=True)

            # Ausgabe des Skripts anzeigen
            print(f"\nOutput for {script_path}:\n")
            if result.stderr:
                print(result.stderr)
            print(result.stdout)
            print("" + "=" * 100)
        else:
            print(f"Script '{script}' not found in directory '{project_root_directory}'")



def create_study():
    stack = inspect.stack()
    caller_file = os.path.splitext(os.path.basename(stack[1].filename))[0]

    storage = RDBStorage(
        url=sqlite_url,
        engine_kwargs={
            'connect_args': {'timeout': 10}
        }
    )

    # Vordefinierte Hyperparameter für LSTM und CNN
    initial_params_normal = {
        'hidden_layer_size': 50,
        'num_layers': 2,
        'batch_size': 16,
        'learn_rate': 0.01,
        'epochs': 50
    }
    initial_params_cnn = {
        'conv1_out_channels': 32,
        'conv2_out_channels': 32,
        'fc1_units': 50,
        'batch_size': 16,
        'learn_rate': 0.01,
        'epochs': 50
    }

    # Bestimme die initialen Parameter basierend auf dem Modelltyp
    if 'cnn' in caller_file:
        initial_params = initial_params_cnn
    else:
        initial_params = initial_params_normal

    study_exists = False
    try:
        study = optuna.load_study(study_name=caller_file, storage=storage)
        study_exists = True
    except KeyError:
        pass

    if not study_exists:
        # Initialize TPESampler
        sampler = optuna.samplers.TPESampler(seed=0, n_startup_trials=10, multivariate=True,
                                             warn_independent_sampling=False)
        study = optuna.create_study(
            study_name=caller_file,
            direction='minimize',
            storage=storage,
            load_if_exists=True,
            sampler=sampler
        )
        # Enqueue the initial parameters for the first trial
        study.enqueue_trial(initial_params)

    print(f"Study '{caller_file}' created or loaded successfully.")
    return study


def delete_study(study_name):
    storage = sqlite_url
    print(f'Deleting study: {study_name}')
    optuna.delete_study(study_name=study_name, storage=storage)
    print(f'Study {study_name} deleted successfully')

def stop_running_study(study_name):
    storage = sqlite_url

    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
        study.stop()
        print(f'Study {study_name} has been stopped.')
    except KeyError:
        print(f'No study found with the name: {study_name}')


def count_studies():
    storage = RDBStorage(url=sqlite_url)
    study_summaries = optuna.get_all_study_summaries(storage=storage)
    num_studies = len(study_summaries)
    print(f"Total number of studies for models: {num_studies}")
    return num_studies


def delete_all_studies():
    storage = sqlite_url
    study_summaries = optuna.get_all_study_summaries(storage=storage)

    for summary in study_summaries:
        study_name = summary.study_name
        optuna.delete_study(study_name=study_name, storage=storage)
        print(f'Study {study_name} deleted successfully')

    print("All studies have been deleted.")


def list_all_studies_with_details():
    storage = sqlite_url
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

    keyword_order = {"lstm": 0, "rnn": 1, "cnn": 2, "gru": 3}
    grouped_studies = {keyword: [] for keyword in keyword_order.keys()}

    for detail in study_details:
        study_name, trial_count, best_value, best_trial_id, params = detail
        for keyword in keyword_order.keys():
            if keyword in study_name:
                grouped_studies[keyword].append(detail)
                break

    current_model_type = None
    for keyword in keyword_order.keys():
        studies = grouped_studies[keyword]
        if not studies:
            continue


        studies.sort(key=lambda x: (x[2] if x[2] is not None else float('inf')))

        if current_model_type != keyword.upper():
            if current_model_type is not None:
                print("-" * 100)
            current_model_type = keyword.upper()
            print(f"{current_model_type} Models:\n")

        for detail in studies:
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

def transfer_trials(source_study_name, target_study_name):
    storage = RDBStorage(url=sqlite_url)

    source_study = optuna.load_study(study_name=source_study_name, storage=storage)
    target_study = optuna.load_study(study_name=target_study_name, storage=storage)

    # Transfer each trial from the source study to the target study
    for trial in source_study.trials:
        target_study.add_trial(trial)

    print(f"Transferred {len(source_study.trials)} trials from study {source_study_name} to study {target_study_name}")


def get_best_trial():
    storage = sqlite_url
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


    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except KeyError:
        print(f'No study found with the name: {study_name}')
        return

    try:
        optuna.delete_trial(study.study_id, trial_id, storage)
        print(f'Trial ID {trial_id} deleted successfully from study {study_name}')
    except KeyError:
        print(f'No trial with ID {trial_id} found in study {study_name}')





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
        return None

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
    print("")

    return best_trial


def get_best_trials_from_study(study_name, n_best=5):
    storage = RDBStorage(url=sqlite_url)
    study = optuna.load_study(study_name=study_name, storage=storage)
    trials = study.trials_dataframe()
    best_trials = trials.sort_values('value').head(n_best)
    return best_trials


def get_all_trials_from_study(study_name):
    storage = sqlite_url

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


def count_all_trials():
    storage = sqlite_url
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
    storage = sqlite_url
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


def get_best_and_worst_trial_from_study(study_name):
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
        return None, None

    best_trial = study.best_trial

    # Find the trial with the highest loss
    trials = study.trials
    if not trials:
        print(f'No trials found for study: {study_name}')
        return best_trial, None

    highest_loss_trial = max(trials, key=lambda t: t.value if t.value is not None else float('-inf'))

    if best_trial:
        print("Best Trial in the " + study_name + " study:")
        print(f"Trial ID: {best_trial.number}")
        print(f"Loss Value: {best_trial.value}")
        print('Best Hyperparameters:')
        for param_name, param_value in best_trial.params.items():
            print(f'  {param_name}: {param_value}')
    else:
        print("No best trial found.")

    if highest_loss_trial:
        print("\nTrial with Highest Loss in the " + study_name + " study:")
        print(f"Trial ID: {highest_loss_trial.number}")
        print(f"Loss Value: {highest_loss_trial.value}")
        print('Hyperparameters:')
        for param_name, param_value in highest_loss_trial.params.items():
            print(f'  {param_name}: {param_value}')
    else:
        print("No trial with highest loss found.")

    return best_trial, highest_loss_trial


def get_trials_with_specific_params(params):
    storage = RDBStorage(
        url=sqlite_url,
        engine_kwargs={
            'connect_args': {'timeout': 10}
        }
    )

    study_summaries = optuna.get_all_study_summaries(storage=storage)
    matching_trials = []

    for study_summary in study_summaries:
        study_name = study_summary.study_name
        study = optuna.load_study(study_name=study_name, storage=storage)
        for trial in study.trials:
            if all(trial.params.get(k) == v for k, v in params.items()):
                matching_trials.append((study_name, trial))

    if matching_trials:
        for study_name, trial in matching_trials:
            print(f"Study: {study_name}")
            print(f"Trial ID: {trial.number}")
            print(f"Loss Value: {trial.value}")
            print("Hyperparameters:")
            for param_name, param_value in trial.params.items():
                print(f"  {param_name}: {param_value}")
            print("" + "-" * 100)
    else:
        print("No matching trials found for the given parameters.")


if __name__ == "__main__":
    models_to_run = [
        # 'gru_standard_all_features_optuna',
        # 'rnn_standard_optuna'
        # 'rnn_standard_all_features_optuna',
        # 'cnn_standard_optuna',
        # 'cnn_standard_all_features_optuna',
        # 'gru_standard_optuna'
        # 'lstm_antiOverfit_optuna',
        # 'lstm_antiOverfit_all_features_optuna',
        # 'lstm_antiOverfit_temp_attention_optuna',
        # 'lstm_standard_all_features_optuna',
        # 'lstm_standard_optuna'
        # 'lstm_temp_attention_all_features_optuna',
        # 'lstm_temp_attention_optuna'
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

    specific_params = {

        'hidden_layer_size': 50,
        'num_layers': 2,
        'batch_size': 16,
        'learn_rate': 0.01,
        'epochs': 50
    }
    print("\nTrials with specific parameters:\n")
    get_trials_with_specific_params(specific_params)

    #get_best_and_worst_trial_from_study("gru_all_features_temp_attention_optuna")


