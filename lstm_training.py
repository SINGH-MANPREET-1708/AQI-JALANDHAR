import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error
)
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, LayerNormalization, Dense,
    TimeDistributed, Add, Dropout, GlobalAveragePooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback


# Custom callback to record learning rate and best validation loss
class RunLogger(Callback):
    def on_train_begin(self, logs=None):
        self.lrs = []
        self.best_val_loss = float('inf')
        self.best_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.lrs.append(float(self.model.optimizer.learning_rate))
        val = logs.get('val_loss')
        if val is not None and val < self.best_val_loss:
            self.best_val_loss = val
            self.best_epoch = epoch + 1


# 1) Load and preprocess data
df = pd.read_csv("jld_aqi_with_aqi.csv", parse_dates=['Date']).set_index('Date')

features = [
    'PM2.5 (ug/m3)', 'PM10 (ug/m3)', 'NO2 (ug/m3)',
    'SO2 (ug/m3)', 'CO (mg/m3)', 'Ozone (ug/m3)'
]

df['AQI'] = df['AQI'].rolling(window=3, min_periods=1).mean()

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features + ['AQI']])
data = pd.DataFrame(scaled, columns=features + ['AQI'], index=df.index)


# 2) Build sequences for time-series training
def build_sequences(arr, window_size=24):
    X, y = [], []
    for i in range(window_size, len(arr)):
        X.append(arr[i - window_size:i, :-1])
        y.append(arr[i, -1])
    return np.array(X), np.array(y)


window_size = 24
X, y = build_sequences(data.values, window_size)

split = int(0.8 * len(X))
Xtr, Xte = X[:split], X[split:]
ytr, yte = y[:split], y[split:]

batch_size = 64


# 3) Define LSTM model architecture
inp = Input(shape=(window_size, len(features)))

x1 = LSTM(128, return_sequences=True)(inp)
n1 = LayerNormalization()(x1)

x2 = LSTM(64, return_sequences=True)(n1)
n2 = LayerNormalization()(x2)

x3 = LSTM(32, return_sequences=True)(n2)
n3 = LayerNormalization()(x3)

p = TimeDistributed(Dense(32))(n1)
r = Add()([p, n3])

d = Dropout(0.2)(r)
f = GlobalAveragePooling1D()(d)
out = Dense(1)(f)

model = Model(inp, out, name="LSTM_Model")
model.compile(optimizer=Adam(1e-3, clipnorm=1.0), loss='mse')


# Model summary info
input_shape = model.input_shape
output_shape = model.output_shape


# 5) Callbacks
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
rlp = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
run_logger = RunLogger()


# 6) Train the model
history = model.fit(
    Xtr, ytr,
    validation_data=(Xte, yte),
    epochs=50,
    batch_size=batch_size,
    callbacks=[es, rlp, run_logger],
    verbose=2
)

stop_epoch = len(history.history['loss'])


# 7) Predict and inverse-scale
y_pred = model.predict(Xte).flatten()


def inv_scale(vals):
    tmp = np.zeros((len(vals), data.shape[1]))
    tmp[:, -1] = vals
    return scaler.inverse_transform(tmp)[:, -1]


y_true = inv_scale(yte)
y_pred_inv = inv_scale(y_pred)


# 8) Performance metrics
mse_ = mean_squared_error(y_true, y_pred_inv)
rmse_ = np.sqrt(mse_)
mae_ = mean_absolute_error(y_true, y_pred_inv)
mape_ = mean_absolute_percentage_error(y_true, y_pred_inv)
r2_ = r2_score(y_true, y_pred_inv)


# 9) Summary report
print("\n=== RUNTIME SUMMARY ===")
print(f"Input shape            : {input_shape}")
print(f"Output shape           : {output_shape}")
print(f"Window size            : {window_size}")
print(f"Batch size             : {batch_size}")
print(f"Stopped at epoch       : {stop_epoch}")
print(f"Best val_loss          : {run_logger.best_val_loss:.6f}")
print(f"Epoch of best val_loss : {run_logger.best_epoch}")

print("\nFinal Test Metrics:")
print(f"  MSE  : {mse_:.2f}")
print(f"  RMSE : {rmse_:.2f}")
print(f"  MAE  : {mae_:.2f}")
print(f"  MAPE : {mape_ * 100:.2f}%")
print(f"  R²   : {r2_:.4f}")


# 10) Diagnostic plots

# (a) Loss curves
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title("Loss Curves")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid()

# (b) Learning rate over epochs
plt.figure(figsize=(8, 4))
plt.plot(run_logger.lrs, label='Learning Rate')
plt.title("Learning Rate over Epochs")
plt.xlabel("Epoch")
plt.ylabel("LR")
plt.legend()
plt.grid()

# (c) Residuals
plt.figure(figsize=(6, 4))
sns.histplot(y_true - y_pred_inv, kde=True, bins=50)
plt.title("Residuals Distribution")
plt.xlabel("Error (AQI)")
plt.ylabel("Frequency")

# (d) Predicted vs Actual
plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred_inv, alpha=0.3)
mn, mx = y_true.min(), y_true.max()
plt.plot([mn, mx], [mn, mx], 'r--')
plt.title("Predicted vs Actual AQI")
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.grid()

# (e) Time-Series comparison
plt.figure(figsize=(8, 4))
plt.plot(y_true, label='Actual AQI')
plt.plot(y_pred_inv, label='Predicted AQI', alpha=0.7)
plt.title("AQI Time Series")
plt.xlabel("Time Index")
plt.ylabel("AQI")
plt.legend()
plt.grid()

# (f) Correlation heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Matrix")

plt.tight_layout()
plt.show()
