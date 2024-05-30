import pandas as pd


def get_data ():
    test_data = pd.read_csv('../data/Crude_Oil_data.csv')
    return test_data