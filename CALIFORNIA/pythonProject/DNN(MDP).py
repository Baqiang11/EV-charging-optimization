import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Load the dataset
data_path = "C:/Users/82344/Desktop/CALIFORNIA/processed_electric_price_filled.csv"
data = pd.read_csv(data_path)

# Feature engineering
data['Total_Renewable_Energy_Production'] = data['Solar Energy Production (kW)'] + data['Wind Energy Production (kW)']
data['Effective_Charging_Capacity'] = data['Charging Station Capacity (kW)'] * (data['EV Charging Efficiency (%)'] / 100)
data['Adjusted_Charging_Demand'] = data['EV Charging Demand (kW)'] * (data['Renewable Energy Usage (%)'] / 100)
data['Net_Energy_Cost'] = data['EV Charging Demand (kW)'] * data['Electricity Price ($/kWh)']
data['Carbon_Footprint_Reduction'] = data['EV Charging Demand (kW)'] * data['Carbon Emissions (kgCO2/kWh)'] * (1 - data['Renewable Energy Usage (%)'] / 100)
data['Renewable_Energy_Efficiency'] = data['Total_Renewable_Energy_Production'] / data['Effective_Charging_Capacity']

# Select features and target variable
features = [
    'Year', 'Month', 'Day', 'DayOfWeek', 'Hour', 'EV Charging Demand (kW)',
    'Solar Energy Production (kW)', 'Wind Energy Production (kW)',
    'Battery Storage (kWh)', 'Charging Station Capacity (kW)',
    'EV Charging Efficiency (%)', 'Number of EVs Charging',
    'Peak Demand (kW)', 'Renewable Energy Usage (%)', 'Grid Stability Index',
    'Carbon Emissions (kgCO2/kWh)', 'Power Outages (hours)', 'Energy Savings ($)',
    'Total_Renewable_Energy_Production', 'Effective_Charging_Capacity',
    'Adjusted_Charging_Demand', 'Net_Energy_Cost', 'Carbon_Footprint_Reduction',
    'Renewable_Energy_Efficiency'
]

X = data[features]
y = data['Electricity Price ($/kWh)']

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Build the DNN model
model = Sequential([
    Dense(64, input_shape=(X_train.shape[1],), activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1)  # Output layer
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the DNN model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Make predictions
y_pred = model.predict(X_test).flatten()

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Calculate MAPE
def calculate_mape(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

mape = calculate_mape(y_test.values.flatten(), y_pred)

print(f'DNN Model - MSE: {mse}, R2: {r2}, MAPE: {mape}%')

# Visualize the actual vs. predicted values
plt.figure(figsize=(14, 7))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
plt.xlabel('Actual Electricity Price')
plt.ylabel('Predicted Electricity Price')
plt.title('Actual vs Predicted Electricity Price - DNN')
plt.tight_layout()
plt.show()

# Visualize the trend of actual and predicted electricity prices over time
y_test_sorted = y_test.reset_index(drop=True)
y_pred_sorted = pd.Series(y_pred).reset_index(drop=True)

plt.figure(figsize=(14, 7))
plt.plot(y_test_sorted, label='Actual Electricity Price')
plt.plot(y_pred_sorted, label='Predicted Electricity Price', alpha=0.7)
plt.xlabel('Time Index')
plt.ylabel('Electricity Price ($/kWh)')
plt.title('Actual vs Predicted Electricity Price Over Time - DNN')
plt.legend()
plt.tight_layout()
plt.show()
