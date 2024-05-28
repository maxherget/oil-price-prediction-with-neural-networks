import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Seeds für Reproduzierbarkeit setzen
np.random.seed(0)

# Daten laden
test_data = pd.read_csv('../data/Crude_Oil_data.csv')
test_data = test_data[['date', 'close']]
test_data['date'] = pd.to_datetime(test_data['date'])

# Manuelle Skalierung der Daten
def min_max_scaling(data):
    min_val = np.min(data)
    max_val = np.max(data)
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data, min_val, max_val

# Manuelle Umkehrung der Skalierung
def inverse_min_max_scaling(scaled_data, min_val, max_val):
    return scaled_data * (max_val - min_val) + min_val

test_data['close'], min_val, max_val = min_max_scaling(test_data['close'])

# Berechnung der Standardabweichung und Vorhersage
def calculate_standard_deviation_predictions(data_frame, n_days):
    data_frame = data_frame.copy()
    data_frame['std_dev'] = data_frame['close'].rolling(window=n_days).std()
    data_frame['prediction'] = data_frame['close'].shift(1) + data_frame['std_dev'].shift(1)
    return data_frame

n_days = 7  # Anzahl der Tage für die Berechnung der Standardabweichung
predicted_data = calculate_standard_deviation_predictions(test_data, n_days)
predicted_data = predicted_data.dropna()

# Skalierung der Vorhersagen zurücksetzen
predicted_data['close'] = inverse_min_max_scaling(predicted_data['close'], min_val, max_val)
predicted_data['prediction'] = inverse_min_max_scaling(predicted_data['prediction'], min_val, max_val)

# Berechnung des MSE Loss
mse_loss = np.mean((predicted_data['close'] - predicted_data['prediction'])**2)
print(f'Mean Squared Error: {mse_loss}')

# Visualisierung
plt.figure(figsize=(14, 5))
plt.plot(predicted_data['date'], predicted_data['close'], label='Actual Prices')
plt.plot(predicted_data['date'], predicted_data['prediction'], label='Predicted Prices', linestyle='dashed')
plt.title('Crude Oil Prices Prediction using Standard Deviation')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()


