import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Initialize the log file
log_file = open("MDP_simulation_log.txt", "w")

def log_message(message):
    log_file.write(message + "\n")
    log_file.flush()

# Load the dataset
data_path = "C:/Users/82344/Desktop/CALIFORNIA/processed_electric_price_with_features.csv"
data = pd.read_csv(data_path)

# Drop the 'Date' and 'Time' columns
data = data.drop(columns=['Date', 'Time'])

route_data_path = "C:/Users/82344/Desktop/CALIFORNIA/route_data_eureka_to_san_diego.csv"
route_data = pd.read_csv(route_data_path)

# Add a row of zeros to the end of the route data
new_row = pd.DataFrame([[0] * len(route_data.columns)], columns=route_data.columns)
route_data = pd.concat([route_data, new_row], ignore_index=True)

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

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Calculate MAPE
def calculate_mape(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mape

mape = calculate_mape(y_test, y_pred)

print(f'XGBoost Model - MSE: {mse}, R2: {r2}, MAPE: {mape}%')

# MDP-related code
def transition(state, action, route_data, data, future_prices):
    soc, hour, current_location_index = state

    P_charge = 11  # kW, charging power
    Useable_Capacity = 100.0  # kWh, usable battery capacity

    # Actual charge per hour (as a percentage of usable capacity)
    Energy_charged_per_hour = P_charge / Useable_Capacity

    # Adjust strategy based on future prices; reduce charging willingness if prices are high
    future_price_penalty = sum(future_prices) / len(future_prices)
    charge_adjustment = max(0, 1 - future_price_penalty / 100)  # Simple price adjustment

    if action == 'Charge' and soc < 1.0:  # While charging
        next_soc = min(soc + Energy_charged_per_hour, 1.0)
        next_location_index = current_location_index  # No location change while charging

    else:
        # Convert soc from percentage to actual energy
        actual_soc_kWh = soc * Useable_Capacity

        # Calculate new energy, subtracting current trip consumption
        new_actual_soc_kWh = max(actual_soc_kWh - route_data.iloc[current_location_index]['Energy_Consumption_kWh'], 0.0)

        # Convert back to percentage form
        next_soc = new_actual_soc_kWh / Useable_Capacity
        next_location_index = min(current_location_index + 1, len(route_data) - 1)  # Change location when driving

    next_hour = (hour + 1) % 24  # Increment hour, keeping within 0-23 range

    return (next_soc, next_hour, next_location_index)

def reward(state, action, next_state, data, route_data, model, future_prices):
    soc, hour, location = state
    next_soc, next_hour, next_location = next_state

    # Parameter settings
    Useable_Capacity = 100.0  # kWh
    charge_efficiency = 3.0  # Lower charging reward weight
    drive_efficiency = 0.5  # Maintain driving reward weight
    drive_penalty = 1.0  # Adjust driving penalty weight
    depletion_penalty = 10.0  # Maintain high penalty for depletion
    distance_factor = route_data.iloc[location]['Distance_m'] / 1000

    # Dynamic factor to avoid extreme values
    soc_factor = 1 / (next_soc + 0.1)

    # Calculate average future price
    future_price_penalty = sum(future_prices) / len(future_prices)

    if action == 'Charge':
        if next_soc < 0.8:  # If soc is below 80%
            reward_charge = soc_factor * charge_efficiency - future_price_penalty
        else:  # Penalize charging above 80% soc
            reward_charge = -charge_efficiency * next_soc
        return reward_charge

    else:  # Reward for driving
        if next_soc <= 0.2:  # If soc is below 20%
            reward_drive = -soc_factor * depletion_penalty  # Strong penalty for low-soc driving
        elif 0.4 <= next_soc <= 0.7:  # If soc is between 40%-70%
            reward_drive = distance_factor * next_soc * drive_efficiency * 1.5  # Enhance mid-soc driving reward
        else:  # If soc is above 20% and not in the middle range
            reward_drive = distance_factor * next_soc * drive_efficiency  # Balanced driving reward
        return reward_drive

# Function to calculate charging cost
def calculate_charging_cost(state, action, next_state, data):
    if action != 'Charge':  # No cost if not charging
        return 0

    soc, hour, _ = state
    next_soc, _, _ = next_state

    # Calculate actual energy charged (kWh)
    energy_charged = (next_soc - soc) * 100.0  # 100kWh is the total battery capacity

    # Avoid negative values (indicating no charging) to prevent logic errors
    if energy_charged <= 0:
        return 0

    # Find the electricity price for the current hour
    current_price = data.loc[data['Hour'] == hour, 'Electricity Price ($/kWh)'].iloc[0]

    # Calculate charging cost
    cost = energy_charged * current_price
    return cost

# Initialize Q-table
initial_soc_values = np.linspace(0.2, 1.0, 5)
hours = range(24)
actions = ['Charge', 'Do Nothing']
Q = {}

for soc in initial_soc_values:
    for hour in hours:
        state = (soc, hour, 0)
        for action in actions:
            Q[(state, action)] = np.random.uniform(0, 1)

alpha = 0.02  # Lower learning rate
gamma = 0.98  # Increase focus on future rewards
epsilon = 0.8  # Higher initial exploration rate to avoid early convergence

# Slightly slower epsilon decay
epsilon_decay = 0.996

# Initialize storage for total charging costs at each SoC
charging_costs_by_soc = {soc: [] for soc in initial_soc_values}
fixed_strategy_costs_by_soc = {soc: [] for soc in initial_soc_values}

# Fixed strategy: Charge whenever the battery isn't full
def fixed_strategy(state, route_data, data, future_prices):
    soc, _, _ = state
    if soc < 1.0:  # Charge if not fully charged
        return 'Charge'
    else:
        return 'Do Nothing'

# Q-learning algorithm
for soc in initial_soc_values:
    for episode in range(100):
        total_reward = 0  # Initialize total reward for each episode
        total_charging_cost = 0  # Initialize total charging cost for each episode
        total_fixed_cost = 0  # Initialize total charging cost for fixed strategy
        initial_location_index = 0
        state = (soc, np.random.choice(hours), initial_location_index)
        epsilon = max(0.01, epsilon * 0.998)

        log_message(f"Episode {episode} with Initial SoC: {soc}")

        low_soc_threshold = 0.2  # Set soc threshold to 20%

        for t in range(100):
            if state[2] == len(route_data) - 1 and action != 'Charge':
                r = reward(state, action, next_state, data, route_data, model, future_prices)

                # Get the old Q-value
                old_q_value = Q[(state, action)]

                # Calculate and update the new Q-value
                Q[(state, action)] = old_q_value + alpha * (
                        r + gamma * max((Q.get((next_state, a), 0) for a in actions)) - old_q_value
                )
                total_reward += r
                log_message(f"Final Reward: {r}, Total Reward: {total_reward}, Steps: {t}")
                log_message(f"Reached destination in {t} steps.")
                break

            if state[0] <= low_soc_threshold:
                action = 'Charge'
            else:
                if np.random.uniform(0, 1) < epsilon:
                    action = np.random.choice(actions)
                else:
                    action = max(actions, key=lambda a: Q.get((state, a), 0))

            future_prices = []
            for i in range(1, 3):
                future_hour = (state[1] + i) % 24
                X_future = data.loc[data['Hour'] == future_hour, features]
                X_future = X_future.iloc[0].values.reshape(1, -1)
                future_price = model.predict(X_future)[0]
                future_prices.append(future_price)

            next_state = transition(state, action, route_data, data, future_prices)

            # Calculate and accumulate charging cost
            charging_cost = calculate_charging_cost(state, action, next_state, data)
            total_charging_cost += charging_cost

            # Calculate charging cost for the fixed strategy
            fixed_action = fixed_strategy(state, route_data, data, future_prices)
            fixed_next_state = transition(state, fixed_action, route_data, data, future_prices)
            fixed_cost = calculate_charging_cost(state, fixed_action, fixed_next_state, data)
            total_fixed_cost += fixed_cost

            Q[(state, action)] = Q.get((state, action), np.random.uniform(0, 1))
            Q[(next_state, action)] = Q.get((next_state, action), np.random.uniform(0, 1))

            r = reward(state, action, next_state, data, route_data, model, future_prices)

            # Update the old Q-value again
            old_q_value = Q[(state, action)]

            # Update using the adjusted expected future Q-value
            expected_future_q = max((Q.get((next_state, a), 0) for a in actions)) - sum(future_prices) / len(future_prices)
            Q[(state, action)] = old_q_value + alpha * (
                    r + gamma * expected_future_q - old_q_value
            )

            log_message(
                f"Episode {episode}, Step {t}: State={state}, Action={action}, Reward={r}, Old Q={old_q_value}, New Q={Q[(state, action)]}"
            )

            state = next_state
            total_reward += r

        charging_costs_by_soc[soc].append(total_charging_cost)
        fixed_strategy_costs_by_soc[soc].append(total_fixed_cost)
        log_message(f"Episode {episode} finished with Total Reward: {total_reward}, Charging Cost: {total_charging_cost}, Fixed Cost: {total_fixed_cost}")

# Plot the total charging cost changes for each SoC
for soc in initial_soc_values:
    plt.figure(figsize=(10, 6))
    if len(charging_costs_by_soc[soc]) > 0:  # Ensure there's data to plot
        plt.plot(range(100), charging_costs_by_soc[soc], label=f'SoC: {int(soc * 100)}% - Q-Learning')
    if len(fixed_strategy_costs_by_soc[soc]) > 0:  # Ensure there's data to plot
        plt.plot(range(100), fixed_strategy_costs_by_soc[soc], label=f'SoC: {int(soc * 100)}% - Fixed Strategy')
    plt.xlabel('Episode')
    plt.ylabel('Total Charging Cost')
    plt.title(f'Total Charging Cost for Initial SoC: {int(soc * 100)}%')
    plt.legend()
    plt.grid(True)
    plt.show()

# Assuming initial_soc_values contains the five initial SoC values you want to calculate
initial_soc_values = np.linspace(0.2, 1.0, 5)  # [0.2, 0.4, 0.6, 0.8, 1.0]
plt.figure(figsize=(10, 6))

for soc_value in initial_soc_values:
    cost_diff = []
    price_gain = []

    # Iterate over each episode to calculate cost differences and price gains
    for q_cost, fixed_cost in zip(charging_costs_by_soc[soc_value], fixed_strategy_costs_by_soc[soc_value]):
        diff = fixed_cost - q_cost  # Cost difference
        gain = (diff / fixed_cost) * 100 if fixed_cost != 0 else 0  # Calculate price gain, avoid division by zero, and multiply by 100% for percentage
        cost_diff.append(diff)
        price_gain.append(gain)

    # Plot the price gain
    plt.plot(range(len(price_gain)), price_gain, label=f'SoC: {int(soc_value * 100)}%')

plt.xlabel('Episode')
plt.ylabel('Price Gain (%)')
plt.title(f'Price Gain for Different Initial SoC Values')
plt.legend()
    plt.grid(True)
    plt.show()

# Close the log file
log_file.close()
