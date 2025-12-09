import optuna
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, LayerNormalization, Dense,
    TimeDistributed, Add, Dropout, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------------------------------------------------
# Load dataset
# -------------------------------------------------------------------
df = pd.read_csv("jld_aqi_with_aqi.csv", parse_dates=['Date'])
df = df.set_index('Date')

features = ['PM2.5 (ug/m3)', 'PM10 (ug/m3)', 'NO2 (ug/m3)',
            'SO2 (ug/m3)', 'CO (mg/m3)', 'Ozone (ug/m3)']

df['AQI'] = df['AQI'].rolling(window=3, min_periods=1).mean()

# Normalization
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features + ['AQI']])
data = pd.DataFrame(scaled, columns=features + ['AQI'], index=df.index)

# -------------------------------------------------------------------
# Build sequences
# -------------------------------------------------------------------
def build_sequences(arr, window):
    X, y = [], []
    for i in range(window, len(arr)):
        X.append(arr[i-window:i, :-1])
        y.append(arr[i, -1])
    return np.array(X), np.array(y)

# -------------------------------------------------------------------
# Optuna Objective Function
# -------------------------------------------------------------------
def objective(trial):

    # Hyperparameters to tune
    window_size = trial.suggest_int("window_size", 12, 48)
    units1      = trial.suggest_int("units_layer1", 32, 128)
    units2      = trial.suggest_int("units_layer2", 16, 64)
    units3      = trial.suggest_int("units_layer3", 8, 32)
    dropout     = trial.suggest_float("dropout", 0.1, 0.5)
    lr          = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size  = trial.suggest_categorical("batch_size", [16, 32, 64])
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "rmsprop"])

    # Prepare sequences
    X, y = build_sequences(data.values, window_size)
    
    split = int(0.8 * len(X))
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]

    # -------------------------------------------------------------------
    # Build LSTM model
    # -------------------------------------------------------------------
    inp = Input(shape=(window_size, len(features)))

    x1 = LSTM(units1, return_sequences=True)(inp)
    n1 = LayerNormalization()(x1)

    x2 = LSTM(units2, return_sequences=True)(n1)
    n2 = LayerNormalization()(x2)

    x3 = LSTM(units3, return_sequences=True)(n2)
    n3 = LayerNormalization()(x3)

    p = TimeDistributed(Dense(units3))(n1)
    r = Add()([p, n3])

    d = Dropout(dropout)(r)
    f = GlobalAveragePooling1D()(d)
    out = Dense(1)(f)

    model = Model(inp, out)

    # Choose optimizer
    optimizer = Adam(learning_rate=lr) if optimizer_name == "adam" else RMSprop(learning_rate=lr)

    model.compile(optimizer=optimizer, loss="mse")

    # -------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------
    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        Xtr, ytr,
        validation_data=(Xte, yte),
        epochs=15,         # Fast tuning 
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )

    # Predict
    pred = model.predict(Xte).flatten()

    # Inverse scaling for only AQI
    temp = np.zeros((len(pred), data.shape[1]))
    temp[:, -1] = pred
    pred_inv = scaler.inverse_transform(temp)[:, -1]

    temp2 = np.zeros((len(yte), data.shape[1]))
    temp2[:, -1] = yte
    yte_inv = scaler.inverse_transform(temp2)[:, -1]

    # Final metric (Optuna minimizes)
    mse = mean_squared_error(yte_inv, pred_inv)
    return mse


# -------------------------------------------------------------------
# Run Optuna Study
# -------------------------------------------------------------------
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=15)

print("\n===================================")
print(" BEST HYPERPARAMETERS FOUND ")
print("===================================\n")
for k, v in study.best_params.items():
    print(f"{k}: {v}")

print("\nBest MSE:", study.best_value)

# Save study results
study.trials_dataframe().to_csv("optuna_lstm_trials.csv", index=False)
