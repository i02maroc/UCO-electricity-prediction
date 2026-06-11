import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

# 1. CONFIGURACIÓN DE RUTAS
DATA_PATH = "../../data/processed/dataset_weather2023.csv"
SAVE_DIR = "../../models/results_data_exploration/" 

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 2. CARGA DE DATOS
print("Cargando dataset meteorológico de Córdoba (2023)...")
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).set_index("timestamp")
df = df.sort_index()

# 3. GENERACIÓN GRÁFICAS
print("Generando gráficos de temperatura y humedad...")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

# --- Gráfica 1: Temperatura (°C) ---
ax1.plot(df.index, df["temp"], color='#e15759', linewidth=0.8, alpha=0.85, label="Temperatura")
ax1.set_title("Evolución Anual de Variables Meteorológicas (Córdoba 2023)", fontsize=14, fontweight='bold', pad=15)
ax1.set_ylabel("Temperatura (°C)", fontsize=11, fontweight='semibold')
ax1.grid(True, linestyle='--', alpha=0.4)
ax1.legend(loc="upper right")

# --- Gráfica 2: Humedad Relativa (%) ---
ax2.plot(df.index, df["rhum"], color='#4e79a7', linewidth=0.8, alpha=0.85, label="Humedad Relativa")
ax2.set_ylabel("Humedad Relativa (%)", fontsize=11, fontweight='semibold')
ax2.set_xlabel("Fecha", fontsize=11, labelpad=10)
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.legend(loc="upper right")

plt.tight_layout()

# 4. GUARDADO
panel_path = os.path.join(SAVE_DIR, "analisis_meteorologico_panel.png")
plt.savefig(panel_path, dpi=300)
print(f"Panel conjunto guardado en: {panel_path}")

plt.figure(figsize=(16, 5))
plt.plot(df.index, df["temp"], color='#e15759', linewidth=0.8, alpha=0.85)
plt.title("Perfil Anual de Temperatura - Córdoba (2023)", fontsize=13, fontweight='bold')
plt.ylabel("Temperatura (°C)")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
temp_path = os.path.join(SAVE_DIR, "meteorologia_temperatura.png")
plt.savefig(temp_path, dpi=300)
plt.close()

plt.figure(figsize=(16, 5))
plt.plot(df.index, df["rhum"], color='#4e79a7', linewidth=0.8, alpha=0.85)
plt.title("Perfil Anual de Humedad Relativa - Córdoba (2023)", fontsize=13, fontweight='bold')
plt.ylabel("Humedad Relativa (%)")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
rhum_path = os.path.join(SAVE_DIR, "meteorologia_humedad.png")
plt.savefig(rhum_path, dpi=300)
plt.close()

print("Gráficas individuales guardadas con éxito.")