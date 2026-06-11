import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import os
warnings.filterwarnings("ignore")

# 1. CARGA DE DATOS (solo necesitamos active_power)
#    NO se aplica dropna() de lags ya que no es necesaria gestionar valores iniciales
df = pd.read_csv("../../data/processed/dataset_electric2023.csv",
                 parse_dates=["timestamp"]).set_index("timestamp")
df = df.sort_index()

serie = df["active_power"].copy()
print(f"Serie cargada: {len(serie)} observaciones horarias")
print(f"Rango: {serie.index[0]} → {serie.index[-1]}")

# 2. DIVISIÓN TEMPORAL — misma que los modelos (70/10/20)
n = len(serie)
train_end = int(n * 0.70)
val_end = int(n * 0.80)

# SARIMA entrena sobre train+val para arrancar justo antes del test, eliminando el gap temporal.
# Sin early stopping ni hiperparámetros que ajustar.
train_val = serie.iloc[:val_end]
test = serie.iloc[val_end:]

print(f"\nTrain+Val: {len(train_val)} obs ({train_val.index[0]} → {train_val.index[-1]})")
print(f"Test:      {len(test)} obs  ({test.index[0]} → {test.index[-1]})")

# 3. MODELO SARIMA(1,1,1)x(1,1,1,24)
#    s=24: estacionalidad diaria (la dominante en series horarias de energía).
#    s=168 (semanal) sería ideal pero es computacionalmente prohibitivo con
#    datos horarios — se documenta como limitación y línea de trabajo futuro.
SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL = (1, 1, 1, 24)

print(f"\nEntrenando SARIMA{SARIMA_ORDER}x{SARIMA_SEASONAL} sobre {len(train_val)} obs...")
print("(Puede tardar varios minutos con datos horarios)")

model_sarima = SARIMAX(
    train_val,
    order=SARIMA_ORDER,
    seasonal_order=SARIMA_SEASONAL,
    enforce_stationarity=False,
    enforce_invertibility=False
)

result_sarima = model_sarima.fit(disp=False)
print("SARIMA entrenado")
print(result_sarima.summary())

# 4. PREDICCIÓN one-shot sobre el set de test (sin data leak)
print(f"\nGenerando predicciones para {len(test)} horas del test...")

y_pred_sarima = result_sarima.forecast(steps=len(test))
y_test_real = test.values
y_naive_real = serie.iloc[val_end - 168 : val_end - 168 + len(test)].values

# 5. GUARDAR PREDICCIONES EN CSV
os.makedirs("../../models/sarima", exist_ok=True)

predictions_df = pd.DataFrame({
    "timestamp":    test.index,
    "real":         y_test_real,
    "sarima_pred":  y_pred_sarima.values,
    "naive_pred":   y_naive_real
})
predictions_df.to_csv("../../models/sarima/sarima_predictions.csv", index=False)
print("Predicciones guardadas en sarima_predictions.csv")

# 6. MÉTRICAS — sobre TODO el test y también sobre la 1ª semana
def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

H = 168
mae_sarima_full = mean_absolute_error(y_test_real, y_pred_sarima)
rmse_sarima_full = np.sqrt(mean_squared_error(y_test_real, y_pred_sarima))
mape_sarima_full = mape(y_test_real, y_pred_sarima)
mae_naive_full = mean_absolute_error(y_test_real, y_naive_real)
mape_naive_full = mape(y_test_real, y_naive_real)

mae_sarima_1w = mean_absolute_error(y_test_real[:H], y_pred_sarima[:H])
rmse_sarima_1w = np.sqrt(mean_squared_error(y_test_real[:H], y_pred_sarima[:H]))
mape_sarima_1w = mape(y_test_real[:H], y_pred_sarima[:H])
mae_naive_1w = mean_absolute_error(y_test_real[:H], y_naive_real[:H])
mape_naive_1w = mape(y_test_real[:H], y_naive_real[:H])

gru_v1 = {"MAE": 50.1, "RMSE": 66.1, "MAPE": 4.12}

print("\n" + "="*70)
print("MÉTRICAS — TEST COMPLETO")
print("="*70)
print(f"{'Métrica':<10} {'Naive':>10} {'SARIMA':>10} {'GRU v1':>10}")
print("="*70)
print(f"{'MAE':<10} {mae_naive_full:>10.1f} {mae_sarima_full:>10.1f} {gru_v1['MAE']:>10.1f}")
print(f"{'MAPE':<10} {mape_naive_full:>9.2f}% {mape_sarima_full:>9.2f}% {gru_v1['MAPE']:>9.2f}%")
print()
print("MÉTRICAS — PRIMERAS 168 HORAS (1 SEMANA)")
print("="*70)
print(f"{'Métrica':<10} {'Naive':>10} {'SARIMA':>10} {'GRU v1':>10}")
print("="*70)
print(f"{'MAE':<10} {mae_naive_1w:>10.1f} {mae_sarima_1w:>10.1f} {gru_v1['MAE']:>10.1f}")
print(f"{'RMSE':<10} {'—':>10} {rmse_sarima_1w:>10.1f} {gru_v1['RMSE']:>10.1f}")
print(f"{'MAPE':<10} {mape_naive_1w:>9.2f}% {mape_sarima_1w:>9.2f}% {gru_v1['MAPE']:>9.2f}%")
print("="*70)

# 7. GRÁFICAS SARIMA STANDALONE
for n_plot, nombre in [(168, "1_semana"), (336, "2_semanas")]:
    test_dates = test.index[:n_plot]
    plt.figure(figsize=(16, 6))
    plt.plot(test_dates, y_test_real[:n_plot], label="Real", linewidth=2, color="#1f77b4")
    plt.plot(test_dates, y_pred_sarima[:n_plot], label="SARIMA", linestyle="--", color="orange", linewidth=2)
    plt.plot(test_dates, y_naive_real[:n_plot], label="Naive (t-168)", alpha=0.5, color="green")
    plt.title(f"SARIMA — Predicción {nombre.replace('_', ' ')} del test")
    plt.xlabel("Fecha"); plt.ylabel("Energía activa (Wh)")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    filename = f"../../models/sarima/prediction_sarima_{nombre}.png"
    plt.savefig(filename, dpi=150); plt.show()
    print(f"Gráfica guardada: {filename}")

# 8. GUARDAR MÉTRICAS EN TXT
with open("../../models/sarima/results_sarima.txt", "w") as f:
    f.write("RESUMEN COMPARATIVO SARIMA VS OTROS\n")
    f.write("="*40 + "\n")
    f.write(f"Entrenado sobre: train+val ({len(train_val)} obs)\n")
    f.write(f"Evaluado sobre:  test ({len(test)} obs)\n\n")
    f.write("TEST COMPLETO:\n")
    f.write(f"  SARIMA MAPE: {mape_sarima_full:.2f}%\n")
    f.write(f"  SARIMA MAE:  {mae_sarima_full:.2f}\n")
    f.write(f"  SARIMA RMSE: {rmse_sarima_full:.2f}\n")
    f.write(f"  NAIVE  MAPE: {mape_naive_full:.2f}%\n\n")
    f.write("PRIMERA SEMANA (168h):\n")
    f.write(f"  SARIMA MAPE: {mape_sarima_1w:.2f}%\n")
    f.write(f"  SARIMA MAE:  {mae_sarima_1w:.2f}\n")
    f.write(f"  SARIMA RMSE: {rmse_sarima_1w:.2f}\n")
    f.write(f"  NAIVE  MAPE: {mape_naive_1w:.2f}%\n")

print("\nMétricas guardadas en results_sarima.txt")