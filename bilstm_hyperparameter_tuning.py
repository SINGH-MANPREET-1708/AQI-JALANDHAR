import optuna
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Bidirectional, LSTM, Dense, Dropout, Add,
    LayerNormalization, TimeDistributed, Attention,
    GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping


# -------------------------------------------------------------------
# Load dataset
# -------------------------------------------------------------------
df = pd.read_csv("jld_aqi_with_aqi.csv", parse_dates=['Date'])
df = df.set_index('Date')

features = [
    'PM2.5 (ug/m3)', 'PM10 (ug/m3)', 'NO2 (ug/m3)',
    'SO2 (ug/m3)', 'CO (mg/m3)', 'Ozone (ug/m3)'
]

df['AQI'] = df['AQI'].rolling(window=3, min_periods=1).mean()

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features + ['AQI']])
data = pd.DataFrame(scaled, columns=features + ['AQI'], index=df.index)


# -------------------------------------------------------------------
# Build sequences
# -------------------------------------------------------------------
def build_sequences(arr, window_size):
    X, y = [], []
    for i in range(window_size, len(arr)):
        X.append(arr[i - window_size:i, :-1])
        y.append(arr[i, -1])
    return np.array(X), np.array(y)


# -------------------------------------------------------------------
# Optuna Objective Function
# -------------------------------------------------------------------
def objective(trial):

    window_size = trial.suggest_int("window_size", 12, 48)
    units1 = trial.suggest_int("units_layer1", 32, 128)
    units2 = trial.suggest_int("units_layer2", 16, 64)
    units3 = trial.suggest_int("units_layer3", 8, 48)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "rmsprop"])

    X, y = build_sequences(data.values, window_size)
    split = int(0.8 * len(X))
    Xtr, Xte = X[:split], X[split:]
    ytr, yte = y[:split], y[split:]

    # ------------------ Model Architecture ------------------
    inp = Input(shape=(window_size, len(features)))

    x1 = Bidirectional(LSTM(units1, return_sequences=True))(inp)
    x1 = LayerNormalization()(x1)

    x2 = Bidirectional(LSTM(units2, return_sequences=True))(x1)
    x2 = LayerNormalization()(x2)

    x3 = Bidirectional(LSTM(units3, return_sequences=True))(x2)
    x3 = LayerNormalization()(x3)

    proj = TimeDistributed(Dense(units3))(x1)
    res = Add()([proj, x3])
    res = Dropout(dropout)(res)

    att = Attention()([res, res])
    att = Dropout(dropout)(att)

    flat = GlobalAveragePooling1D()(att)
    out = Dense(1)(flat)

    model = Model(inputs=inp, outputs=out)

    optimizer = Adam(lr) if optimizer_name == "adam" else RMSprop(lr)
    model.compile(optimizer=optimizer, loss="mse")

    # ------------------ Training ------------------
    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(
        Xtr, ytr,
        validation_data=(Xte, yte),
        epochs=15,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )

    # ------------------ Inverse Scaling ------------------
    pred = model.predict(Xte).flatten()

    temp = np.zeros((len(pred), data.shape[1]))
    temp[:, -1] = pred
    pred_inv = scaler.inverse_transform(temp)[:, -1]

    temp2 = np.zeros((len(yte), data.shape[1]))
    temp2[:, -1] = yte
    yte_inv = scaler.inverse_transform(temp2)[:, -1]

    mse = mean_squared_error(yte_inv, pred_inv)
    return mse


# -------------------------------------------------------------------
# Run Study
# -------------------------------------------------------------------
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=20)

print("\n==============================")
print(" BEST BiLSTM HYPERPARAMETERS ")
print("==============================\n")

for k, v in study.best_params.items():
    print(f"{k}: {v}")

print("\nBest MSE:", study.best_value)

study.trials_dataframe().to_csv("optuna_bilstm_trials.csv", index=False)
