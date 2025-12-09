import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, LayerNormalization, Dense, TimeDistributed,
    Add, Dropout, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam


# ---------------------------------------------------------------------
# 1. Load and preprocess data
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# 2. Build sequences for supervised learning
# ---------------------------------------------------------------------
def build_sequences(arr, window_size=24):
    X, y = [], []
    for i in range(window_size, len(arr)):
        X.append(arr[i - window_size:i, :-1])
        y.append(arr[i, -1])
    return np.array(X), np.array(y)


window_size = 24
X, y = build_sequences(data.values, window_size)

# Split dataset into 3 simulated clients
split1 = int(0.33 * len(X))
split2 = int(0.66 * len(X))

client_data = [
    (X[:split1], y[:split1]),
    (X[split1:split2], y[split1:split2]),
    (X[split2:], y[split2:])
]


# ---------------------------------------------------------------------
# 3. Define LSTM model architecture
# ---------------------------------------------------------------------
def create_model():
    inp = Input(shape=(window_size, len(features)))

    x1 = LSTM(128, return_sequences=True)(inp)
    n1 = LayerNormalization()(x1)

    x2 = LSTM(64, return_sequences=True)(n1)
    n2 = LayerNormalization()(x2)

    x3 = LSTM(32, return_sequences=True)(n2)
    n3 = LayerNormalization()(x3)

    proj = TimeDistributed(Dense(32))(n1)
    res = Add()([proj, n3])
    res = Dropout(0.2)(res)

    flat = GlobalAveragePooling1D()(res)
    out = Dense(1)(flat)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse')

    return model


# ---------------------------------------------------------------------
# 4. Federated Training (FedAvg)
# ---------------------------------------------------------------------
global_model = create_model()

rounds = 5
local_epochs = 3

history_dict = {"round": [], "loss": []}

for rnd in range(rounds):
    print(f"\n--- Round {rnd + 1} ---")

    local_weights = []
    local_losses = []

    for X_client, y_client in client_data:
        local_model = create_model()
        local_model.set_weights(global_model.get_weights())

        history = local_model.fit(
            X_client, y_client,
            epochs=local_epochs,
            batch_size=64,
            verbose=0
        )

        local_weights.append(local_model.get_weights())
        local_losses.append(history.history['loss'][-1])

    # FedAvg aggregation
    new_weights = [
        np.mean([client_weights[layer] for client_weights in local_weights], axis=0)
        for layer in range(len(local_weights[0]))
    ]

    global_model.set_weights(new_weights)

    avg_loss = np.mean(local_losses)
    history_dict["round"].append(rnd + 1)
    history_dict["loss"].append(avg_loss)

    print(f"Average Loss: {avg_loss:.5f}")


# ---------------------------------------------------------------------
# 5. Final Evaluation on Combined Dataset
# ---------------------------------------------------------------------
X_all = np.concatenate([cd[0] for cd in client_data])
y_all = np.concatenate([cd[1] for cd in client_data])

y_pred_scaled = global_model.predict(X_all).flatten()

def inverse_scale(values):
    temp = np.zeros((len(values), data.shape[1]))
    temp[:, -1] = values
    return scaler.inverse_transform(temp)[:, -1]

y_pred = inverse_scale(y_pred_scaled)
y_true = inverse_scale(y_all)

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\n=== Final Global Model Evaluation ===")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ---------------------------------------------------------------------
# 6. Plots
# ---------------------------------------------------------------------
plt.figure(figsize=(14, 5))

# Loss per FL round
plt.subplot(1, 2, 1)
plt.plot(history_dict["round"], history_dict["loss"], marker='o')
plt.title("Federated Learning: Global Loss per Round")
plt.xlabel("Round")
plt.ylabel("Loss (MSE)")
plt.grid(True)

# Actual vs Predicted
plt.subplot(1, 2, 2)
plt.plot(y_true[:200], label='Actual AQI')
plt.plot(y_pred[:200], label='Predicted AQI', linestyle='--')
plt.title("Actual vs Predicted AQI")
plt.xlabel("Sample Index")
plt.ylabel("AQI")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
