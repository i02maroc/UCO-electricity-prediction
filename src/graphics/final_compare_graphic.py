import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import os

warnings.filterwarnings("ignore")

# 1. RUTAS Y CONFIGURACIÓN
MODEL_DIR_V4 = "../../models/v4/gru/1111/"
MODEL_DIR_V5 = "../../models/v5/gru/2222/"
SARIMA_CSV   = "../../models/sarima/sarima_predictions.csv"
SAVE_DIR     = "../../models/results_final_comparativa/"
os.makedirs(SAVE_DIR, exist_ok=True)

N_PAST = 168

# 2. CARGA DE DATOS BASE (compartida por ambos modelos)
print("Cargando datos...")
df_electric = pd.read_csv("../../data/processed/dataset_electric2023.csv", parse_dates=["timestamp"]).set_index("timestamp")
df_academic = pd.read_csv("../../data/processed/dataset_academic2023.csv", parse_dates=["timestamp"]).set_index("timestamp")
df_weather  = pd.read_csv("../../data/processed/dataset_weather2023.csv",  parse_dates=["timestamp"]).set_index("timestamp")

df_base = df_electric.join(df_academic, how="inner").join(df_weather, how="inner").sort_index()

# Variables cíclicas compartidas
df_base["hour_sin"]  = np.sin(2 * np.pi * df_base.index.hour / 24)
df_base["hour_cos"]  = np.cos(2 * np.pi * df_base.index.hour / 24)
df_base["dow_sin"]   = np.sin(2 * np.pi * df_base.index.dayofweek / 7)
df_base["dow_cos"]   = np.cos(2 * np.pi * df_base.index.dayofweek / 7)
df_base["month_sin"] = np.sin(2 * np.pi * df_base.index.month / 12)
df_base["month_cos"] = np.cos(2 * np.pi * df_base.index.month / 12)

# 3. PREDICCIÓN GRU V4
print("Preparando features V4...")
df_v4 = df_base.copy()
df_v4["power_t-24"]        = df_v4["active_power"].shift(24)
df_v4["power_t-168"]       = df_v4["active_power"].shift(168)
df_v4["target_is_holiday"] = df_v4["is_holiday"].shift(-1)
df_v4 = df_v4.dropna()

feature_cols_v4 = [
    "active_power", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    "month_sin", "month_cos", "is_weekend", "is_class", "is_exam",
    "is_admin", "temp", "power_t-24", "power_t-168", "target_is_holiday"
]
data_v4    = df_v4[feature_cols_v4].copy()
n_v4       = len(data_v4)
val_end_v4 = int(n_v4 * 0.80)

scaler_target_v4   = joblib.load(MODEL_DIR_V4 + "scaler_target_v4.pkl")
scaler_features_v4 = joblib.load(MODEL_DIR_V4 + "scaler_features_v4.pkl")
model_v4           = load_model(MODEL_DIR_V4 + "gru_model_v4.keras")

target_scaled_v4   = scaler_target_v4.transform(data_v4[["active_power"]].values)
features_scaled_v4 = scaler_features_v4.transform(data_v4[feature_cols_v4[1:]].values)
data_scaled_v4     = np.hstack([target_scaled_v4, features_scaled_v4])

def create_sequences(arr, seq_length):
    return np.array([arr[i:i + seq_length] for i in range(len(arr) - seq_length)])

X_test_v4       = create_sequences(data_scaled_v4, N_PAST)[val_end_v4:]
preds_v4_scaled = model_v4.predict(X_test_v4, verbose=0)
y_pred_v4       = scaler_target_v4.inverse_transform(preds_v4_scaled).flatten()

test_start_v4  = val_end_v4 + N_PAST
gru_test_dates = df_v4.index[test_start_v4 : test_start_v4 + len(y_pred_v4)]
y_real         = df_v4["active_power"].values[test_start_v4 : test_start_v4 + len(y_pred_v4)]

print(f"GRU V4 — {len(y_pred_v4)} obs ({gru_test_dates[0]} → {gru_test_dates[-1]})")


# 4. PREDICCIÓN GRU V5
print("Preparando features V5...")
df_v5 = df_base.copy()
df_v5["is_weekend"] = (df_v5.index.dayofweek >= 5).astype(float)

feature_cols_v5 = [
    "active_power",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend"
]
data_v5    = df_v5[feature_cols_v5].copy()
n_v5       = len(data_v5)
val_end_v5 = int(n_v5 * 0.80)

scaler_target_v5   = joblib.load(MODEL_DIR_V5 + "scaler_target_v5.pkl")
scaler_features_v5 = joblib.load(MODEL_DIR_V5 + "scaler_features_v5.pkl")
model_v5           = load_model(MODEL_DIR_V5 + "gru_model_v5.keras")

target_scaled_v5   = scaler_target_v5.transform(data_v5[["active_power"]].values)
features_scaled_v5 = scaler_features_v5.transform(data_v5[feature_cols_v5[1:]].values)
data_scaled_v5     = np.hstack([target_scaled_v5, features_scaled_v5])

X_test_v5        = create_sequences(data_scaled_v5, N_PAST)[val_end_v5:]
preds_v5_scaled  = model_v5.predict(X_test_v5, verbose=0)
y_pred_v5_full   = scaler_target_v5.inverse_transform(preds_v5_scaled).flatten()

test_start_v5 = val_end_v5 + N_PAST
v5_test_dates = df_v5.index[test_start_v5 : test_start_v5 + len(y_pred_v5_full)]

# Alineación por timestamp contra el índice de referencia V4
y_pred_v5 = pd.Series(y_pred_v5_full, index=v5_test_dates).reindex(gru_test_dates).values

print(f"GRU V5 — {len(y_pred_v5_full)} obs ({v5_test_dates[0]} → {v5_test_dates[-1]})")
if np.isnan(y_pred_v5).any():
    print(f"⚠️  V5: {np.isnan(y_pred_v5).sum()} timestamps sin alinear con V4")

# 5. CARGA DE PREDICCIONES SARIMA DESDE CSV
print("Cargando predicciones SARIMA desde CSV...")
sarima_df      = pd.read_csv(SARIMA_CSV, parse_dates=["timestamp"]).set_index("timestamp")
sarima_aligned = sarima_df.reindex(gru_test_dates)

if sarima_aligned["sarima_pred"].isna().any():
    print(f"SARIMA: {sarima_aligned['sarima_pred'].isna().sum()} timestamps no encontrados.")
    print("   Ejecuta sarima.py antes de este script.")

y_pred_sarima = sarima_aligned["sarima_pred"].values
y_naive       = sarima_aligned["naive_pred"].values
test_dates    = gru_test_dates

print(f"SARIMA alineado — {len(y_pred_sarima)} obs")

# 6. MÉTRICAS COMPARATIVAS
def mape(y_true, y_pred):
    mask = (y_true != 0) & ~np.isnan(y_pred)
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

for nombre_h, lim in [("1 Semana (168h)", 168), ("Test completo", len(y_real))]:
    mae_v4     = mean_absolute_error(y_real[:lim], y_pred_v4[:lim])
    mae_v5     = mean_absolute_error(y_real[:lim], y_pred_v5[:lim])
    mae_sarima = mean_absolute_error(y_real[:lim], y_pred_sarima[:lim])
    mae_naive  = mean_absolute_error(y_real[:lim], y_naive[:lim])

    rmse_v4     = np.sqrt(mean_squared_error(y_real[:lim], y_pred_v4[:lim]))
    rmse_v5     = np.sqrt(mean_squared_error(y_real[:lim], y_pred_v5[:lim]))
    rmse_sarima = np.sqrt(mean_squared_error(y_real[:lim], y_pred_sarima[:lim]))

    mape_v4     = mape(y_real[:lim], y_pred_v4[:lim])
    mape_v5     = mape(y_real[:lim], y_pred_v5[:lim])
    mape_sarima = mape(y_real[:lim], y_pred_sarima[:lim])
    mape_naive  = mape(y_real[:lim], y_naive[:lim])

    print(f"\n{'='*78}")
    print(f"  MÉTRICAS — {nombre_h}")
    print(f"{'='*78}")
    print(f"{'Métrica':<10} {'Naive':>10} {'SARIMA':>10} {'GRU V5':>10} {'GRU V4':>10}")
    print(f"{'='*78}")
    print(f"{'MAE':<10} {mae_naive:>10.1f} {mae_sarima:>10.1f} {mae_v5:>10.1f} {mae_v4:>10.1f}")
    print(f"{'RMSE':<10} {'—':>10} {rmse_sarima:>10.1f} {rmse_v5:>10.1f} {rmse_v4:>10.1f}")
    print(f"{'MAPE':<10} {mape_naive:>9.2f}% {mape_sarima:>9.2f}% {mape_v5:>9.2f}% {mape_v4:>9.2f}%")
    print(f"{'='*78}")

# 7. GRÁFICAS COMPARATIVAS
horizontes = {"1_Semana": 168, "2_Semanas": 336, "Completo": len(test_dates)}

for nombre, limit in horizontes.items():
    fig, ax = plt.subplots(figsize=(18, 8))

    ax.plot(test_dates[:limit], y_real[:limit],
            label="Real (Consumo)", color="black", linewidth=1.5, zorder=5)

    ax.plot(test_dates[:limit], y_naive[:limit],
            label="Naive (t-168)", color="#2ca02c",
            linestyle=":", alpha=0.6, zorder=1)

    ax.plot(test_dates[:limit], y_pred_sarima[:limit],
            label="SARIMA (Baseline Estadístico)", color="#ff7f0e",
            linestyle="--", alpha=0.8, zorder=2)

    ax.plot(test_dates[:limit], y_pred_v5[:limit],
            label="GRU V5 (features básicas)", color="#1f77b4",
            linestyle="--", linewidth=2, zorder=3)

    ax.plot(test_dates[:limit], y_pred_v4[:limit],
            label="GRU V4 (propuesto)", color="#9467bd",
            linestyle="--", linewidth=2.5, zorder=4)

    ax.set_title(f"Comparativa Final de Modelos — Vista {nombre.replace('_', ' ')}",
                 fontsize=15, fontweight="bold")
    ax.set_ylabel("Energía Activa (Wh)", fontsize=12)
    ax.set_xlabel("Fecha", fontsize=12)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    plt.tight_layout()
    out_path = f"{SAVE_DIR}macro_comparativa_{nombre}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Gráfica guardada: {out_path}")

print(f"\nProceso finalizado. Gráficas guardadas en: {SAVE_DIR}")