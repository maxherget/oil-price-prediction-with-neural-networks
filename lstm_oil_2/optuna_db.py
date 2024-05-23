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
    print(f'{study_name} gets deleted from DB')
    # Verbindet sich mit der SQLite-Datenbank
    conn = sqlite3.connect('optuna_study.db')
    cursor = conn.cursor()

    # Schritt 1: Löschen der zugehörigen Bewertungswerte (trial_values)
    cursor.execute('''
        DELETE FROM trial_values
        WHERE trial_id IN (
            SELECT trial_id
            FROM trials
            WHERE study_id = (
                SELECT study_id
                FROM studies
                WHERE study_name = ?
            )
        )
    ''', (study_name,))

    # Schritt 2: Löschen der zugehörigen Hyperparameter (trial_params)
    cursor.execute('''
        DELETE FROM trial_params
        WHERE trial_id IN (
            SELECT trial_id
            FROM trials
            WHERE study_id = (
                SELECT study_id
                FROM studies
                WHERE study_name = ?
            )
        )
    ''', (study_name,))

    # Schritt 3: Löschen der zugehörigen Versuche (trials)
    cursor.execute('''
        DELETE FROM trials
        WHERE study_id = (
            SELECT study_id
            FROM studies
            WHERE study_name = ?
        )
    ''', (study_name,))

    # Schritt 4: Jetzt die Studie selbst löschen
    cursor.execute('''
        DELETE FROM studies
        WHERE study_name = ?
    ''', (study_name,))

    # Änderungen speichern und Verbindung schließen
    conn.commit()
    conn.close()

def find_best_trial():
    # Verbindet sich mit der SQLite-Datenbank
    conn = sqlite3.connect('optuna_study.db')
    cursor = conn.cursor()

    # Finden des Trials mit dem geringsten Loss-Wert
    cursor.execute('''
        SELECT trials.study_id, trials.trial_id, trials.value, studies.study_name
        FROM trials
        JOIN studies ON trials.study_id = studies.study_id
        WHERE trials.value IS NOT NULL
        ORDER BY trials.value ASC
        LIMIT 1
    ''')
    best_trial = cursor.fetchone()

    if best_trial:
        study_id, trial_id, value, study_name = best_trial
        print(f'Best Trial ID: {trial_id}')
        print(f'Loss Value: {value}')
        print(f'Study ID: {study_id}')
        print(f'Study Name: {study_name}')

        # Finden der Hyperparameter für den besten Trial
        cursor.execute('''
            SELECT param_name, param_value
            FROM trial_params
            WHERE trial_id = ?
        ''', (trial_id,))
        params = cursor.fetchall()

        print('Hyperparameters:')
        for param in params:
            param_name, param_value = param
            print(f'  {param_name}: {param_value}')
    else:
        print('No trials found.')

    # Verbindung schließen
    conn.close()


def find_best_trial_in_study(study_name):
    # Verbindet sich mit der SQLite-Datenbank
    conn = sqlite3.connect('optuna_study.db')
    cursor = conn.cursor()

    # Ermitteln der study_id für den gegebenen Studiennamen
    cursor.execute('''
        SELECT study_id
        FROM studies
        WHERE study_name = ?
    ''', (study_name,))
    result = cursor.fetchone()

    if not result:
        print(f'No study found with the name: {study_name}')
        conn.close()
        return

    study_id = result[0]

    # Finden des besten Trials in der angegebenen Studie
    cursor.execute('''
        SELECT trial_id, value
        FROM trials
        WHERE study_id = ?
        AND value IS NOT NULL
        ORDER BY value ASC
        LIMIT 1
    ''', (study_id,))
    best_trial = cursor.fetchone()

    if best_trial:
        trial_id, value = best_trial
        print(f'Best Trial ID: {trial_id}')
        print(f'Loss Value: {value}')
        print(f'Study Name: {study_name}')

        # Finden der Hyperparameter für den besten Trial
        cursor.execute('''
            SELECT param_name, param_value
            FROM trial_params
            WHERE trial_id = ?
        ''', (trial_id,))
        params = cursor.fetchall()

        print('Hyperparameters:')
        for param in params:
            param_name, param_value = param
            print(f'  {param_name}: {param_value}')
    else:
        print(f'No trials found for study: {study_name}')

    # Verbindung schließen
    conn.close()




if __name__ == "__main__":
    find_best_trial()
    # delete_study('')
    # find_best_trial_in_study('')

