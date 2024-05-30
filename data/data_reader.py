import pandas as pd
import os


def get_data():
    # Bestimme den absoluten Pfad zur Datendatei relativ zum Speicherort dieses Skripts
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', 'Crude_Oil_data.csv')
    test_data = pd.read_csv(data_path)
    return test_data
