# EV Charging Optimization with Machine Learning and Reinforcement Learning

This project implements **intelligent charging strategies for Electric Vehicles (EVs)** by combining **electricity price forecasting** with **Markov Decision Process (MDP)** and **Q-learning**.  
The system demonstrates how agent-based optimization can minimize EV charging cost under dynamic electricity prices.

---

## 📌 Project Structure

├── XGBoost.py # Electricity price forecasting with XGBoost

├── random forest.py # Electricity price forecasting with Random Forest

├── DNN(MDP).py # Electricity price forecasting with Deep Neural Network (DNN) only

├── DNN_model.py # Full pipeline: DNN forecasting + MDP/Q-learning optimization

├── XGBoost(MDP).py # XGBoost forecasting + MDP/Q-learning optimization

---

## ⚡ Features

- **Price Forecasting Models**
  - `XGBoost.py`: Forecast electricity prices with XGBoost
  - `random forest.py`: Forecast electricity prices with Random Forest
  - `DNN(MDP).py`: Forecast electricity prices with Deep Neural Network (DNN)
- **Optimization with Agentic Systems**
  - `XGBoost(MDP).py`: XGBoost forecasting integrated with MDP + Q-learning
  - `DNN_model.py`: DNN forecasting integrated with MDP + Q-learning
- **Reinforcement Learning Framework**
  - States: State-of-Charge (SoC), Hour, Location index
  - Actions: `Charge` or `Do Nothing`
  - Reward: Combines SoC level, driving distance, and electricity price
  - Learning: Q-learning with ε-greedy exploration
  - Baseline: Fixed strategy (always charge when not full)


---
## 📊 Data Requirements

- **Electricity Price Data**
  - CSV file containing:
    - Target column: `Electricity Price ($/kWh)`
    - Time features: `Year, Month, Day, DayOfWeek, Hour`
    - Other engineered features (e.g., renewable energy, load)
- **Route Data**
  - CSV file containing:
    - `Energy_Consumption_kWh`: energy required for each segment
    - `Distance_m`: distance of each route segment

Modify the file paths in scripts to match your dataset locations.



---
## 🚀 How to Run

### 1. Price Prediction Only
```bash
python XGBoost.py
python random\ forest.py
python DNN(MDP).py
Outputs:

Regression metrics: MSE, R², MAPE

Figures:

Actual vs Predicted scatter plot

Price prediction trends over time

---

---
### 2. Full Optimization (Price Prediction + MDP/Q-learning)

python XGBoost(MDP).py
python DNN_model.py
---
Outputs:

Q-learning vs Fixed Strategy comparison

Total charging cost curves (per initial SoC)

Price Gain(%) vs Episode plots

Simulation logs (optional)

📈 Example Results
Price Forecasting (XGBoost/DNN/RF)

R² typically between 0.85–0.95 depending on features

Optimization Results

Q-learning agent converges to policies that reduce total charging cost

Up to 20–40% Price Gain compared to the fixed charging strategy

⚙️ Dependencies
Python 3.8+

Required packages:

pip install numpy pandas matplotlib scikit-learn xgboost tensorflow
🔮 Future Improvements
Generalize predictor module to support plug-and-play (XGBoost / RF / DNN / others).

Replace hardcoded paths with argparse or YAML config.

Add multi-station, queueing, and power-capacity constraints.

Improve reproducibility: random seed control, unified logging, and saved models.

📝 Citation
If you use this work, please cite or acknowledge:

Ziyu Zheng, Optimization of Intelligent Charging Strategy for Electric Vehicles based on Machine Learning, Reinforcement Learning, and Markov Decision Process, University of Sussex, 2024.

---
