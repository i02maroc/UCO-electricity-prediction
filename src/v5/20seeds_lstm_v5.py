import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import joblib
import os

# 1 Y 2. CARGA DE DATOS Y FEATURE ENGINEERING
print("Cargando y preparando datos (V5 - Base V1 + Optimizaciones de Entrenamiento)...")
df = pd.read_csv("../../data/processed/dataset_electric2023.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)
df = df.sort_index()

# Features idénticas al V1 (NO se añaden lags ni variables externas)
df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
df["dow_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
df["dow_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
df["is_weekend"] = (df.index.dayofweek >= 5).astype(float)

feature_cols = [
    "active_power",
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin","month_cos",
    "is_weekend"
]

data = df[feature_cols].copy()

# 3. ESCALADO — Scaler separado para target y features
scaler_target = MinMaxScaler()
scaler_features = MinMaxScaler()

target_scaled = scaler_target.fit_transform(data[["active_power"]].values)
features_scaled = scaler_features.fit_transform(data[feature_cols[1:]].values)
data_scaled = np.hstack([target_scaled, features_scaled])

# 4. CREACIÓN DE SECUENCIAS
SEQ_LENGTH = 168 # 7 días
TARGET_IDX = 0

def create_sequences(data, seq_length, target_idx):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, target_idx])
    return np.array(X), np.array(y)

X, y = create_sequences(data_scaled, SEQ_LENGTH, TARGET_IDX)

# 5. DIVISIÓN TEMPORAL — 70% train | 10% val | 20% test
n = len(X)
train_end = int(n * 0.70)
val_end  = int(n * 0.80)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test  = X[val_end:], y[val_end:]

# Preparación para Naive y gráficas
test_start_idx = val_end + SEQ_LENGTH
y_test_real = scaler_target.inverse_transform(y_test.reshape(-1,1)).flatten()
y_naive_real = df["active_power"].values[test_start_idx-168 : test_start_idx-168+len(y_test_real)]

def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

mape_naive = mape(y_test_real, y_naive_real)

# Referencias para comparativa
mape_v1_medio = 3.99

# BUCLE DE 20 SEMILLAS CON GUARDADO
SEEDS = [40, 121, 800, 3123, 46291, 1234, 5555, 9999, 42, 777,
         10101, 2024, 8888, 3333, 1111, 2222, 6666, 7654, 9876, 54321]

resultados_mape = []

print("\n" + "="*65)
print(f"INICIANDO EJECUCIÓN LSTM V5 CON GUARDADO")
print(f" Arquitectura: Igual que V1")
print(f" Mejoras: LR inicial 5e-4 + ReduceLROnPlateau + Patience=30 + 150 épocas")
print(f" Baseline Naive: {mape_naive:.2f}% | Referencia V1: {mape_v1_medio:.2f}%")
print("="*65 + "\n")

for i, SEED in enumerate(SEEDS):
    # 1. Configuración de Semilla y Carpetas
    os.environ['PYTHONHASHSEED'] = str(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    save_path = f"../../models/v5/lstm/{SEED}/"
    os.makedirs(save_path, exist_ok=True)

    # 2. Construir Modelo
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=(SEQ_LENGTH, X.shape[2])),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1)
    ])

    # Mejora 1: LR inicial más fino (5e-4 en vez del 1e-3 por defecto)
    # El Adam con LR=1e-3 converge rápido pero puede saltarse
    # el mínimo global. Con 5e-4 el descenso es más preciso desde el inicio.
    optimizer = Adam(learning_rate=5e-4)
    model.compile(optimizer=optimizer, loss="mae")

    # Mejora 2: ReduceLROnPlateau
    # Cuando la val_loss se estanca 8 épocas seguidas, reduce el LR
    # a la mitad (factor=0.5). Permite seguir refinando sin sobreajuste.
    # El LR mínimo es 1e-6 para evitar que se paralice completamente.
    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-6,
        verbose=0
    )

    # Mejora 3: Patience=30 y épocas=150
    # Con 30 el modelo tiene margen para recuperarse de mesetas temporales en la val_loss.
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True
    )

    # 3. Entrenamiento
    history = model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop, lr_scheduler],
        verbose=0
    )

    # 4. Predicción y métricas
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred_real = scaler_target.inverse_transform(y_pred_scaled).flatten()

    mae_val = mean_absolute_error(y_test_real, y_pred_real)
    rmse_val = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    mape_val = mape(y_test_real, y_pred_real)
    resultados_mape.append(mape_val)

    epocas_reales = len(history.history['loss'])

    # 5. Guardar Modelo, Scalers y Métricas
    model.save(f"{save_path}lstm_model_v5.keras")
    joblib.dump(scaler_target, f"{save_path}scaler_target_v5.pkl")
    joblib.dump(scaler_features, f"{save_path}scaler_features_v5.pkl")

    # Auditoría de errores (Top 15 peores)
    errores_absolutos = np.abs(y_test_real - y_pred_real)
    test_index = df.index[test_start_idx : test_start_idx + len(y_test_real)]
    df_errores = pd.DataFrame({
        'Real (Wh)': y_test_real.round(1),
        'Prediccion (Wh)': y_pred_real.round(1),
        'Error Absoluto': errores_absolutos.round(1)
    }, index=test_index)
    top_errores = df_errores.sort_values(by='Error Absoluto', ascending=False).head(15)

    with open(f"{save_path}metrics.txt", "w") as f:
        f.write(f"SEED: {SEED}\n")
        f.write(f"MAE: {mae_val:.2f}\n")
        f.write(f"RMSE: {rmse_val:.2f}\n")
        f.write(f"MAPE: {mape_val:.2f}%\n")
        f.write(f"Epocas entrenadas: {epocas_reales}\n\n")
        f.write("TOP 15 HORAS CON MAYOR ERROR (AUDITORÍA V5):\n")
        f.write(top_errores.to_string())
        f.write("\n")

    # 6. GENERAR Y GUARDAR GRÁFICAS
    # GRÁFICA 1: CURVA DE APRENDIZAJE
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Val loss')
    plt.title(f'LSTM V5 — Semilla {SEED} — Curva de Aprendizaje ({epocas_reales} épocas)')
    plt.xlabel('Época'); plt.ylabel('MAE (escalado)')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path}learning_curve_v5.png", dpi=100)
    plt.close()

    # GRÁFICA 2: PREDICCIÓN 1 SEMANA (168h)
    n_plot_1 = 168
    t_idx_1 = df.index[test_start_idx : test_start_idx + n_plot_1]
    plt.figure(figsize=(12, 6))
    plt.plot(t_idx_1, y_test_real[:n_plot_1], label='Real', linewidth=2, color='#1f77b4')
    plt.plot(t_idx_1, y_pred_real[:n_plot_1], label='LSTM V5', linestyle='--', color='#0d742f')
    plt.title(f'LSTM V5 — Semilla {SEED} — Predicción 1ª Semana (MAPE: {mape_val:.2f}%)')
    plt.ylabel('Energía activa (Wh)')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path}prediction_1week_v5.png", dpi=100)
    plt.close()

    # GRÁFICA 3: PREDICCIÓN 2 SEMANAS (336h)
    n_plot_2 = 336
    t_idx_2 = df.index[test_start_idx : test_start_idx + n_plot_2]
    plt.figure(figsize=(15, 6))
    plt.plot(t_idx_2, y_test_real[:n_plot_2], label='Real', linewidth=2, color='#1f77b4')
    plt.plot(t_idx_2, y_pred_real[:n_plot_2], label='LSTM V5', linestyle='--', color='#0d742f')
    plt.title(f'LSTM V5 — Semilla {SEED} — Predicción General (2 Semanas)')
    plt.ylabel('Energía activa (Wh)')
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{save_path}prediction_2weeks_v5.png", dpi=100)
    plt.close()

    print(f"Iteración {i+1:02d}/20 | Semilla: {SEED:<6} | MAPE: {mape_val:.2f}% | Épocas: {epocas_reales:3d} | Guardado V5")
    tf.keras.backend.clear_session()

# RESUMEN
print("\n" + "="*65)
print(f"RESULTADO GLOBAL ESTADÍSTICO LSTM V5:")
print(f"  MAPE Medio:    {np.mean(resultados_mape):.2f}% ± {np.std(resultados_mape):.2f}%")
print(f"  Mejor semilla: {min(resultados_mape):.2f}%")
print(f"  Peor semilla:  {max(resultados_mape):.2f}%")
print("="*65)
print(f"\nCOMPARATIVA:")
print(f"  LSTM V1 (referencia): {mape_v1_medio:.2f}%")
print(f"  LSTM V5 (este run):   {np.mean(resultados_mape):.2f}%")
print("="*65 + "\n")