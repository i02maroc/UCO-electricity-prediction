import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

# 1. CARGA DE DATOS
df = pd.read_csv("../../data/processed/dataset_electric2023.csv",
                 parse_dates=["timestamp"]).set_index("timestamp")
df = df.sort_index()

serie = df["active_power"].copy()
print(f"Serie cargada: {len(serie)} observaciones horarias")
print(f"Rango: {serie.index[0]} → {serie.index[-1]}")

# 2. DIVISIÓN TEMPORAL (70/10/20)
n         = len(serie)
train_end = int(n * 0.70)
val_end   = int(n * 0.80)

test       = serie.iloc[val_end:]
test_dates = test.index
y_real     = test.values

# Naive: misma hora de la semana anterior (t-168h)
y_naive = serie.iloc[val_end - 168 : val_end - 168 + len(test)].values

print(f"\nTest:  {len(test)} obs  ({test_dates[0]} → {test_dates[-1]})")
print(f"Naive: desplazamiento de 168h (1 semana atrás)")

# 3. MÉTRICAS
def mape(y_true, y_pred):
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

H = 168

mae_full  = mean_absolute_error(y_real, y_naive)
rmse_full = np.sqrt(mean_squared_error(y_real, y_naive))
mape_full = mape(y_real, y_naive)

mae_1w  = mean_absolute_error(y_real[:H], y_naive[:H])
rmse_1w = np.sqrt(mean_squared_error(y_real[:H], y_naive[:H]))
mape_1w = mape(y_real[:H], y_naive[:H])

print("\n" + "="*50)
print("MÉTRICAS NAIVE — TEST COMPLETO")
print("="*50)
print(f"  MAE:  {mae_full:.2f} Wh")
print(f"  RMSE: {rmse_full:.2f} Wh")
print(f"  MAPE: {mape_full:.2f}%")
print()
print("MÉTRICAS NAIVE — PRIMERA SEMANA (168h)")
print("="*50)
print(f"  MAE:  {mae_1w:.2f} Wh")
print(f"  RMSE: {rmse_1w:.2f} Wh")
print(f"  MAPE: {mape_1w:.2f}%")
print("="*50)

# 4. GRÁFICAS
os.makedirs("../../models/naive", exist_ok=True)

for n_plot, nombre in [(168, "1_semana"), (336, "2_semanas"), (len(test), "completo")]:
    fig, ax = plt.subplots(figsize=(16, 6))

    ax.plot(test_dates[:n_plot], y_real[:n_plot],
            label="Real", linewidth=1.5, color="black")
    ax.plot(test_dates[:n_plot], y_naive[:n_plot],
            label="Naive (t-168)", linestyle="--", color="#2ca02c", linewidth=2)

    ax.set_title(f"Baseline Naive — Vista {nombre.replace('_', ' ')} del test",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Fecha", fontsize=12)
    ax.set_ylabel("Energía Activa (Wh)", fontsize=12)
    ax.legend(loc="upper right", frameon=True, fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_path = f"../../models/naive/prediction_naive_{nombre}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Gráfica guardada: {out_path}")

# 5. GUARDAR MÉTRICAS EN TXT
with open("../../models/naive/results_naive.txt", "w") as f:
    f.write("RESULTADOS BASELINE NAIVE (t-168)\n")
    f.write("="*40 + "\n")
    f.write(f"Evaluado sobre: test ({len(test)} obs)\n\n")
    f.write("TEST COMPLETO:\n")
    f.write(f"  MAE:  {mae_full:.2f} Wh\n")
    f.write(f"  RMSE: {rmse_full:.2f} Wh\n")
    f.write(f"  MAPE: {mape_full:.2f}%\n\n")
    f.write("PRIMERA SEMANA (168h):\n")
    f.write(f"  MAE:  {mae_1w:.2f} Wh\n")
    f.write(f"  RMSE: {rmse_1w:.2f} Wh\n")
    f.write(f"  MAPE: {mape_1w:.2f}%\n")

print("\nMétricas guardadas en results_naive.txt")