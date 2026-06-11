import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import joblib
import os
import random

# Semillas de prueba iniciales
SEED = 40
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 1. CARGA DE DATOS
df_electric = pd.read_csv("../../data/processed/dataset_electric2023.csv", parse_dates=["timestamp"]).set_index("timestamp")
df_academic = pd.read_csv("../../data/processed/dataset_academic2023.csv", parse_dates=["timestamp"]).set_index("timestamp")
df_weather = pd.read_csv("../../data/processed/dataset_weather2023.csv", parse_dates=["timestamp"]).set_index("timestamp")

# Unir por timestamp
df = df_electric.join(df_academic, how="inner").join(df_weather, how="inner")
df = df.sort_index()

print(f"Dataset combinado: {len(df)} filas")
print(f"Rango: {df.index[0]} → {df.index[-1]}")

# 2. FEATURES
df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

# Lags explícitos (El "atajo" de memoria para la GRU)
df["power_t-24"] = df["active_power"].shift(24)
df["power_t-168"] = df["active_power"].shift(168)

# Visión de futuro. Le damos a la red la capacidad de saber si mañana es festivo
df["target_is_holiday"] = df["is_holiday"].shift(-1)

# Limpieza de NaNs por desplazamientos
df = df.dropna()

feature_cols = [
    "active_power",    # target
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "is_weekend",

    "is_class",
    "is_exam",
    "is_admin",

    "temp",

    "power_t-24",
    "power_t-168",
    "target_is_holiday"
]

data = df[feature_cols].copy()

# 3. ESCALADO — Scaler separado para target y features
scaler_target = MinMaxScaler()
scaler_features = MinMaxScaler()

target_scaled = scaler_target.fit_transform(data[["active_power"]].values)
features_scaled = scaler_features.fit_transform(data[feature_cols[1:]].values)
data_scaled = np.hstack([target_scaled, features_scaled])

# 4. CREACIÓN DE SECUENCIAS
SEQ_LENGTH = 168  # 7 días
TARGET_IDX = 0

def create_sequences(data: np.ndarray, seq_length: int, target_idx: int):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, target_idx])
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, SEQ_LENGTH, TARGET_IDX)
print(f"\nSecuencias: X={X.shape}, y={y.shape}")

# 5. DIVISIÓN TEMPORAL — 70% train | 10% val | 20% test
n = len(X)
train_end = int(n * 0.70)
val_end = int(n * 0.80)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

# 6. MODELO GRU
model = Sequential([
    GRU(128, return_sequences=True, input_shape=(SEQ_LENGTH, X.shape[2])),
    Dropout(0.2),
    GRU(64, return_sequences=False),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1)
], name="gru_v4")

model.compile(optimizer="adam", loss="mae")
model.summary()

# 7. ENTRENAMIENTO
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=20,
    restore_best_weights=True,
    verbose=1
)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stop],
    verbose=1
)

# 8. GUARDARDADO
os.makedirs(f"../../models/v4/gru/{SEED}/", exist_ok=True)
model.save(f"../../models/v4/gru/{SEED}/gru_model_v4.keras")
joblib.dump(scaler_target, f"../../models/v4/gru/{SEED}/scaler_target_gru_v4.pkl")
joblib.dump(scaler_features, f"../../models/v4/gru/{SEED}/scaler_features_gru_v4.pkl")
print("\n✓ Modelo y scalers guardados")

# 9. PREDICCIÓN E INVERSIÓN DE ESCALA
y_pred_scaled = model.predict(X_test)
y_test_real = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_real = scaler_target.inverse_transform(y_pred_scaled).flatten()

# 10. BASELINE NAIVE — (t-168)
test_start_idx = val_end + SEQ_LENGTH
y_naive_real = df["active_power"].values[
    test_start_idx - 168 : test_start_idx - 168 + len(y_test_real)
]

# 11. MÉTRICAS
def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

mae_v4 = mean_absolute_error(y_test_real, y_pred_real)
rmse_v4 = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
mape_v4 = mape(y_test_real, y_pred_real)

mae_naive = mean_absolute_error(y_test_real, y_naive_real)
rmse_naive = np.sqrt(mean_squared_error(y_test_real, y_naive_real))
mape_naive = mape(y_test_real, y_naive_real)

# Referencias hardcoded de versiones anteriores
mae_v1, rmse_v1, mape_v1 = 55.8, 67.7, 5.10
mae_v2, rmse_v2, mape_v2 = 55.1, 69.7, 5.04

print("\n" + "="*65)
print(f"{'Métrica':<10} {'gru v1':>10} {'gru v2':>10} {'gru v4':>10} {'Naive':>10}")
print("="*65)
print(f"{'MAE':<10} {mae_v1:>10.1f} {mae_v2:>10.1f} {mae_v4:>10.1f} {mae_naive:>10.1f}")
print(f"{'RMSE':<10} {rmse_v1:>10.1f} {rmse_v2:>10.1f} {rmse_v4:>10.1f} {rmse_naive:>10.1f}")
print(f"{'MAPE':<10} {mape_v1:>9.2f}% {mape_v2:>9.2f}% {mape_v4:>9.2f}% {mape_naive:>9.2f}%")
print("="*65)

# Auditoría de errores (Top 15 peores)
errores_absolutos = np.abs(y_test_real - y_pred_real)
test_index = df.index[test_start_idx : test_start_idx + len(y_test_real)]

df_errores = pd.DataFrame({
    'Real (Wh)': y_test_real.round(1),
    'Prediccion (Wh)': y_pred_real.round(1),
    'Error Absoluto': errores_absolutos.round(1)
}, index=test_index)

top_errores = df_errores.sort_values(by='Error Absoluto', ascending=False).head(15)

print("\n" + "="*20)
print("TOP 15 HORAS CON MAYOR ERROR EN EL TEST")
print("="*20)
print(top_errores)

# 12. GRÁFICAS
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Curva de pérdida durante entrenamiento
axes[0].plot(history.history["loss"], label="Train loss")
axes[0].plot(history.history["val_loss"], label="Val loss")
axes[0].set_title("GRU v4 — Curva de aprendizaje")
axes[0].set_xlabel("Época")
axes[0].set_ylabel("MAE (escalado)")
axes[0].legend()
axes[0].grid(True)

# Predicciones vs real (todo el test de validación)
n_plot = len(y_test_real)
test_index = df.index[test_start_idx : test_start_idx + n_plot]
axes[1].plot(test_index, y_test_real[:n_plot], label="Real", linewidth=2)
axes[1].plot(test_index, y_pred_real[:n_plot], label="gru v4", linestyle="--")
axes[1].plot(test_index, y_naive_real[:n_plot], label="Naive (t-168)", alpha=0.6)
axes[1].set_title("gru v4 — Predicción Completa del Set de Test")
axes[1].set_xlabel("Fecha")
axes[1].set_ylabel("Energía activa (Wh)")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(f"../../models/v4/gru/{SEED}/prediction_gru_v4.png", dpi=150)
plt.show()
print("Gráfica guardada")