import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    mean_absolute_percentage_error, r2_score
)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Bidirectional, LSTM, Dense, Dropout, Add, LayerNormalization,
    TimeDistributed, Attention, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback


# ---------------------------------------------------------------------
# Custom Callback for tracking LR and best validation epoch
# ---------------------------------------------------------------------
class RunLogger(Callback):
    def on_train_begin(self, logs=None):
        self.lrs = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.lrs.append(float(self.model.optimizer.learning_rate.numpy()))
        val_loss = logs.get('val_loss')
        if val_loss is not None and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch + 1


# ---------------------------------------------------------------------
# 1. Load and preprocess data
# ---------------------------------------------------------------------
df = pd.read_csv("jld_aqi_with_aqi.csv", parse_dates=['Date'])
df = df.set_index('Date')

features = [
    'PM2.5 (ug/m3)', 'PM10 (ug/m3)', 'NO2 (ug/m3)',
    'SO2 (ug/m3)', 'CO (mg/m3)', 'Ozone (ug/m3)'
]

# Smooth AQI slightly
df['AQI'] = df['AQI'].rolling(window=3, min_periods=1).mean()

# Scaling
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features + ['AQI']])
data = pd.DataFrame(scaled, columns=features + ['AQI'], index=df.index)


# ---------------------------------------------------------------------
# 2. Create sequences
# ---------------------------------------------------------------------
def build_sequences(arr, window_size):
    X, y = [], []
    for i in range(window_size, len(arr)):
        X.append(arr[i-window_size:i, :-1])
        y.append(arr[i, -1])
    return np.array(X), np.array(y)


window_size = 24
X, y = build_sequences(data.values, window_size)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

batch_size = 64


# ---------------------------------------------------------------------
# 3. BiLSTM + Attention Model
# ---------------------------------------------------------------------
inp = Input(shape=(window_size, len(features)))

x1 = Bidirectional(LSTM(128, return_sequences=True))(inp)
x1 = LayerNormalization()(x1)

x2 = Bidirectional(LSTM(64, return_sequences=True))(x1)
x2 = LayerNormalization()(x2)

x3 = Bidirectional(LSTM(32, return_sequences=True))(x2)
x3 = LayerNormalization()(x3)

proj = TimeDistributed(Dense(64))(x1)
residual = Add()([proj, x3])
residual = Dropout(0.2)(residual)

att = Attention()([residual, residual])
att = Dropout(0.2)(att)

flat = GlobalAveragePooling1D()(att)
out = Dense(1)(flat)

model = Model(inputs=inp, outputs=out)
model.compile(optimizer=Adam(learning_rate=1e-3, clipnorm=1.0), loss='mse')


# ---------------------------------------------------------------------
# 4. Callbacks
# ---------------------------------------------------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
logger = RunLogger()


# ---------------------------------------------------------------------
# 5. Training
# ---------------------------------------------------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=batch_size,
    callbacks=[early_stop, reduce_lr, logger],
    verbose=2
)

stop_epoch = len(history.history['loss'])


# ---------------------------------------------------------------------
# 6. Inverse scaling
# ---------------------------------------------------------------------
def inverse_scale(values):
    temp = np.zeros((len(values), data.shape[1]))
    temp[:, -1] = values
    return scaler.inverse_transform(temp)[:, -1]


y_pred = model.predict(X_test).flatten()
y_pred_inv = inverse_scale(y_pred)
y_true_inv = inverse_scale(y_test)


# ---------------------------------------------------------------------
# 7. Metrics
# ---------------------------------------------------------------------
mse = mean_squared_error(y_true_inv, y_pred_inv)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true_inv, y_pred_inv)
mape = mean_absolute_percentage_error(y_true_inv, y_pred_inv)
r2 = r2_score(y_true_inv, y_pred_inv)


# ---------------------------------------------------------------------
# 8. Runtime Summary
# ---------------------------------------------------------------------
print("\n=== RUNTIME SUMMARY ===")
print(f"Input shape              : {model.input_shape}")
print(f"Output shape             : {model.output_shape}")
print(f"Window size              : {window_size}")
print(f"Batch size               : {batch_size}")
print(f"Stopped at epoch         : {stop_epoch}")
print(f"Best val_loss            : {logger.best_val_loss:.6f}")
print(f"Best epoch               : {logger.best_epoch}")

print("\nFinal Metrics:")
print(f"  MSE   : {mse:.3f}")
print(f"  RMSE  : {rmse:.3f}")
print(f"  MAE   : {mae:.3f}")
print(f"  MAPE  : {mape*100:.2f}%")
print(f"  R²    : {r2:.4f}")


# ---------------------------------------------------------------------
# 9. Plots
# ---------------------------------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss Curves")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid()
plt.legend()

plt.figure(figsize=(8, 4))
plt.plot(logger.lrs, label='Learning Rate')
plt.title("Learning Rate Schedule")
plt.xlabel("Epoch")
plt.ylabel("LR")
plt.grid()
plt.legend()

plt.figure(figsize=(6, 4))
sns.histplot(y_true_inv - y_pred_inv, bins=50, kde=True)
plt.title("Residuals Distribution")
plt.xlabel("Error (AQI)")
plt.ylabel("Frequency")

plt.figure(figsize=(6, 6))
plt.scatter(y_true_inv, y_pred_inv, alpha=0.3)
min_val, max_val = y_true_inv.min(), y_true_inv.max()
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.title("Predicted vs Actual AQI")
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.grid()

plt.figure(figsize=(8, 4))
plt.plot(y_true_inv, label='Actual AQI')
plt.plot(y_pred_inv, label='Predicted AQI')
plt.title("AQI Time Series")
plt.xlabel("Time Index")
plt.ylabel("AQI")
plt.grid()
plt.legend()

plt.figure(figsize=(6, 5))
sns.heatmap(data.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")

plt.tight_layout()
plt.show()
