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
SEED = 800
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 1. CARGA DE DATOS
df_electric = pd.read_csv("../../data/processed/dataset_electric2023.csv",
                            parse_dates=["timestamp"]).set_index("timestamp")
df_academic = pd.read_csv("../../data/processed/dataset_academic2023.csv",
                            parse_dates=["timestamp"]).set_index("timestamp")

# Unir por timestamp
df = df_electric.join(df_academic, how="inner")
df = df.sort_index()

print(f"Dataset combinado: {len(df)} filas")
print(f"Rango: {df.index[0]} → {df.index[-1]}")
print(f"Columnas: {list(df.columns)}")

# 2. FEATURES
df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

feature_cols = [
    "active_power",   # target
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "is_weekend",

    "is_class",  
    "is_exam",   
    "is_admin", 
    # day_type==0 queda implícito cuando las tres anteriores son 0
]

data = df[feature_cols].copy()
print(f"\nFeatures ({len(feature_cols)}): {feature_cols}")
print(f"NaN en dataset: {data.isna().sum().sum()}")

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
print(f"\nSecuencias generadas: X={X.shape}, y={y.shape}")

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
], name="gru_v2")

model.compile(optimizer="adam", loss="mae")
model.summary()

# 7. ENTRENAMIENTO
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
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
os.makedirs(f"../../models/v2/gru/{SEED}/", exist_ok=True)
model.save(f"../../models/v2/gru/{SEED}/gru_model_v2.keras")
joblib.dump(scaler_target, f"../../models/v2/gru/{SEED}/scaler_target_gru_v2.pkl")
joblib.dump(scaler_features, f"../../models/v2/gru/{SEED}/scaler_features_gru_v2.pkl")
print("\nModelo GRU y scalers guardados")

# 9. PREDICCIÓN E INVERSIÓN DE ESCALA
y_pred_scaled = model.predict(X_test)
y_test_real = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_real = scaler_target.inverse_transform(y_pred_scaled).flatten()

# 10. BASELINE NAIVE — (t-168)
test_start_idx = val_end + SEQ_LENGTH
y_naive_real = df["active_power"].values[
    test_start_idx - 168 : test_start_idx - 168 + len(y_test_real)
]

print(f"\nGRU   — primeras 5 predicciones: {y_pred_real[:5].round(1)}")
print(f"Real   — primeras 5 valores:      {y_test_real[:5].round(1)}")
print(f"Naive  — primeras 5 predicciones: {y_naive_real[:5].round(1)}")

# 11. MÉTRICAS
def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

mae_v2 = mean_absolute_error(y_test_real, y_pred_real)
rmse_v2 = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
mape_v2 = mape(y_test_real, y_pred_real)

mae_naive = mean_absolute_error(y_test_real, y_naive_real)
rmse_naive = np.sqrt(mean_squared_error(y_test_real, y_naive_real))
mape_naive = mape(y_test_real, y_naive_real)

# Resultados v1 de referencia (hardcoded de la ejecución anterior)
mae_v1, rmse_v1, mape_v1 = 44.1, 59.2, 3.82

print("\n" + "="*58)
print(f"{'Métrica':<10} {'gru v1':>10} {'gru v2':>10} {'Naive':>12}")
print("="*58)
print(f"{'MAE':<10} {mae_v1:>10.1f} {mae_v2:>10.1f} {mae_naive:>12.1f}")
print(f"{'RMSE':<10} {rmse_v1:>10.1f} {rmse_v2:>10.1f} {rmse_naive:>12.1f}")
print(f"{'MAPE':<10} {mape_v1:>9.2f}% {mape_v2:>9.2f}% {mape_naive:>11.2f}%")
print("="*58)

print("\n" + "="*50)
print(f"Semillas generadas:")
print(f"    Semilla de Numpy: {SEED}")
print(f"    Semilla de TensorFlow: {SEED}")
print("="*50 + "\n")

# 12. GRÁFICAS
fig, axes = plt.subplots(2, 1, figsize=(16, 10))

# Curva de aprendizaje
axes[0].plot(history.history["loss"], label="Train loss")
axes[0].plot(history.history["val_loss"], label="Val loss")
axes[0].set_title("GRU v2 — Curva de aprendizaje")
axes[0].set_xlabel("Época")
axes[0].set_ylabel("MAE (escalado)")
axes[0].legend()
axes[0].grid(True)

# Predicciones vs real (primera semana del test)
n_plot = 168
test_index = df.index[test_start_idx : test_start_idx + n_plot]
axes[1].plot(test_index, y_test_real[:n_plot], label="Real", linewidth=2)
axes[1].plot(test_index, y_pred_real[:n_plot], label="gru v2", linestyle="--")
axes[1].plot(test_index, y_naive_real[:n_plot], label="Naive (t-168)", alpha=0.6)
axes[1].set_title("GRU v2 — Predicción primera semana del test")
axes[1].set_xlabel("Fecha")
axes[1].set_ylabel("Energía activa (Wh)")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig(f"../../models/v2/gru/{SEED}/prediction_gru_v2.png", dpi=150)
plt.show()
print(f"Gráfica guardada")