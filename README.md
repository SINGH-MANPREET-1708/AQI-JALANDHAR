# AQI Forecasting using Deep Learning and Federated Learning

This repository contains end-to-end code for Air Quality Index (AQI) forecasting using multiple deep learning approaches, including LSTM, BiLSTM with Attention, and a simulated Federated Learning (FedAvg) setup.

The project includes data preprocessing, AQI computation (CPCB standard), hyperparameter tuning using Optuna, centralized model training, and federated training.

## Project Structure

AQI-JALANDHAR/
│
├── preprocess.py # Timestamp fixing, interpolation, data cleaning
├── compute_aqi.py # AQI calculation (CPCB sub-indices + final AQI)
│
├── lstm_hyperparameter_tuning.py # Optuna tuning for LSTM model
├── lstm_training.py # Final LSTM training using tuned parameters
│
├── bilstm_hyperparameter_tuning.py # Optuna tuning for BiLSTM model
├── bilstm_training.py # Final BiLSTM training using tuned parameters
│
├── federated_approach.py # Simulated Federated Averaging for AQI prediction
│
└── requirements.txt # Dependencies


##  Features

- Complete AQI forecasting pipeline
- Preprocessing with interpolation + timestamp correction
- CPCB-based AQI sub-index computation
- LSTM and BiLSTM models with:
  - Attention
  - Layer Normalization
  - Dropout
  - Residual connections
- Hyperparameter tuning using Optuna
- Federated Learning (3 simulated clients using FedAvg)
- Evaluation metrics:
  - MAE, MSE, RMSE
  - MAPE
  - R² score
- Visualization:
  - Loss curves
  - Predicted vs Actual
  - Residuals
  - Correlation heatmap
  - Learning rate plots


## Installation

Clone the repository:

git clone https://github.com/SINGH-MANPREET-1708/AQI-JALANDHAR.git

cd AQI-JALANDHAR

## Install dependencies:

pip install -r requirements.txt

## Data Preprocessing

Step 1: Clean timestamps + fill missing hours

python preprocess.py

Outputs:

jld_aqi_filled.csv

Step 2: Compute AQI using CPCB breakpoints

python compute_aqi.py

Outputs:

jld_aqi_with_aqi.csv

## LSTM Model

Hyperparameter tuning

python lstm_hyperparameter_tuning.py

Training

python lstm_training.py

## BiLSTM + Attention Model

Hyperparameter tuning

python bilstm_hyperparameter_tuning.py

Training

python bilstm_training.py

## Federated Learning (FedAvg)

Run simulated FL with 3 clients:

python federated_approach.py

Performs:

Local client training

Weighted model averaging

Global model evaluation

## Requirements

numpy==1.23.5
pandas==1.5.3
matplotlib==3.7.1
seaborn==0.12.2
scikit-learn==1.2.2
tensorflow==2.12.0
optuna==3.2.0

## Contact
Manpreet Singh
B.Tech CSE (AI & ML)
DAV Institute of Engineering & Technology, Jalandhar
Email: mrsingh31524@gmail.com
+91 62806-20692

