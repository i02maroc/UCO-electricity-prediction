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

# 1 Y 2. CARGA Y FEATURE ENGINEERING
print("Cargando datos y aplicando Ingeniería de Variables V4...")
df_electric = pd.read_csv("../../data/processed/dataset_electric2023.csv", parse_dates=["timestamp"]).set_index("timestamp")
df_academic = pd.read_csv("../../data/processed/dataset_academic2023.csv", parse_dates=["timestamp"]).set_index("timestamp")
df_weather = pd.read_csv("../../data/processed/dataset_weather2023.csv", parse_dates=["timestamp"]).set_index("timestamp")

# Unir por timestamp
df = df_electric.join(df_academic, how="inner").join(df_weather, how="inner").sort_index()

df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)

# Lags (Memoria) y Visión de Futuro (Holiday Shift)
df["power_t-24"] = df["active_power"].shift(24)
df["power_t-168"] = df["active_power"].shift(168)
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

# 3. ESCALADO Y SECUENCIAS
scaler_target = MinMaxScaler()
scaler_features = MinMaxScaler()

target_scaled = scaler_target.fit_transform(data[["active_power"]].values)
features_scaled = scaler_features.fit_transform(data[feature_cols[1:]].values)
data_scaled = np.hstack([target_scaled, features_scaled])

SEQ_LENGTH = 168
TARGET_IDX = 0

def create_sequences(data, seq_length, target_idx):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, target_idx])
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, SEQ_LENGTH, TARGET_IDX)

n = len(X)
train_end = int(n * 0.70)
val_end = int(n * 0.80)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

# Preparación para Naive y gráficas
test_start_idx = val_end + SEQ_LENGTH
y_test_real = scaler_target.inverse_transform(y_test.reshape(-1,1)).flatten()
y_naive_real = df["active_power"].values[test_start_idx-168 : test_start_idx-168+len(y_test_real)]

def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

mape_naive = mape(y_test_real, y_naive_real)

# BUCLE DE 20 SEMILLAS CON GUARDADO
SEEDS = [40, 121, 800, 3123, 46291, 1234, 5555, 9999, 42, 777, 
         10101, 2024, 8888, 3333, 1111, 2222, 6666, 7654, 9876, 54321]

resultados_mape = []

print("\n" + "="*60)
print(f"INICIANDO EJECUCIÓN GRU V4 CON GUARDADO")
print(f" Baseline Naive: {mape_naive:.2f}%")
print("="*60 + "\n")

for i, SEED in enumerate(SEEDS):
    # 1. Configuración de Semilla y Carpetas
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    
    save_path = f"../../models/v4/gru/{SEED}/"
    os.makedirs(save_path, exist_ok=True)
    
    # 2. Construir Modelo
    model = Sequential([
        GRU(128, return_sequences=True, input_shape=(SEQ_LENGTH, X.shape[2])),
        Dropout(0.2),
        GRU(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ], name="gru_v4")
    model.compile(optimizer="adam", loss="mae")
    
    # 3. Entrenamiento
    early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, 
                        validation_data=(X_val, y_val), callbacks=[early_stop], verbose=0)
    
    # 4. Predicción y Métricas
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred_real = scaler_target.inverse_transform(y_pred_scaled).flatten()
    
    mae_val = mean_absolute_error(y_test_real, y_pred_real)
    rmse_val = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    mape_val = mape(y_test_real, y_pred_real)
    resultados_mape.append(mape_val)
    
    # 5. Guardar Modelo, Scalers y Métricas
    model.save(f"{save_path}gru_model_v4.keras")
    joblib.dump(scaler_target, f"{save_path}scaler_target_v4.pkl")
    joblib.dump(scaler_features, f"{save_path}scaler_features_v4.pkl")
    
    # Auditoría de errores (Top 15 peores)
    errores_abs = np.abs(y_test_real - y_pred_real)
    test_dates = df.index[test_start_idx : test_start_idx + len(y_test_real)]
    top_15_idx = np.argsort(errores_abs)[-15:][::-1]
    
    with open(f"{save_path}metrics.txt", "w") as f:
        f.write(f"SEED: {SEED}\nMAE: {mae_val:.2f}\nRMSE: {rmse_val:.2f}\nMAPE: {mape_val:.2f}%\n\n")
        f.write("TOP 15 PEORES PREDICCIONES:\n")
        f.write(f"{'Fecha':<20} | {'Real':>10} | {'Pred':>10} | {'Error':>10}\n")
        for idx in top_15_idx:
            f.write(f"{str(test_dates[idx]):<20} | {y_test_real[idx]:>10.1f} | {y_pred_real[idx]:>10.1f} | {errores_abs[idx]:>10.1f}\n")

    # 6. GENERAR Y GUARDAR GRÁFICAS
    # GRÁFICA 1: CURVA DE APRENDIZAJE
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train loss'); plt.plot(history.history['val_loss'], label='Val loss')
    plt.title(f'GRU V4 - Semilla {SEED} - Curva de Aprendizaje'); plt.legend(); plt.grid(True)
    plt.savefig(f"{save_path}learning_curve_v4.png", dpi=100); plt.close()

    # GRÁFICA 2: PREDICCIÓN 1 SEMANA (168h)
    plt.figure(figsize=(12, 6))
    plt.plot(test_dates[:168], y_test_real[:168], label='Real', color='#1f77b4')
    plt.plot(test_dates[:168], y_pred_real[:168], label='GRU V4', linestyle='--', color='#9467bd')
    plt.title(f'GRU V4 - Semilla {SEED} - Predicción 1ª Semana'); plt.legend(); plt.grid(True)
    plt.savefig(f"{save_path}prediction_1week_v4.png", dpi=100); plt.close()

    # GRÁFICA 3: PREDICCIÓN 2 SEMANAS (336h)
    plt.figure(figsize=(15, 6))
    plt.plot(test_dates[:336], y_test_real[:336], label='Real', color='#1f77b4')
    plt.plot(test_dates[:336], y_pred_real[:336], label='GRU V4', linestyle='--', color='#9467bd')
    plt.title(f'GRU V4 - Semilla {SEED} - Predicción 2 Semanas'); plt.legend(); plt.grid(True)
    plt.savefig(f"{save_path}prediction_2weeks_v4.png", dpi=100); plt.close()

    print(f"Iteración {i+1:02d}/20 | Semilla: {SEED:<6} | MAPE: {mape_val:.2f}% | Guardado V4")
    tf.keras.backend.clear_session()

# RESUMEN
print("\n" + "="*20)
print(f"RESULTADO GLOBAL ESTADÍSTICO GRU V4:")
print(f"MAPE Medio: {np.mean(resultados_mape):.2f}% ± {np.std(resultados_mape):.2f}%")
print(f"Mejor: {min(resultados_mape):.2f}% | Peor: {max(resultados_mape):.2f}%")
print("="*20 + "\n")