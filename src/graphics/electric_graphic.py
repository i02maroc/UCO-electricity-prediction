import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

# 1. CONFIGURACIÓN DE RUTAS
DATA_PATH = "../../data/processed/dataset_electric2023.csv"
SAVE_DIR = "../../models/results_data_exploration/"

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


# 2. CARGA Y PREPARACIÓN DE DATOS
print("Cargando dataset eléctrico de la Universidad de Córdoba...")
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).set_index("timestamp")
df = df.sort_index()

active_col = "active_power"
reactive_col = "reactive_power" if "reactive_power" in df.columns else None

# 3. GRÁFICO 1: CONSUMO ACTIVO (POTENCIA ACTIVA - Wh)
print("Generando gráfica de Consumo Activo...")
plt.figure(figsize=(16, 6))
plt.plot(df.index, df[active_col], color='#1f77b4', linewidth=1, alpha=0.85)

plt.title("Perfil Anual de Consumo Eléctrico Activo (2023)", fontsize=14, fontweight='bold')
plt.xlabel("Fecha", fontsize=11)
plt.ylabel("Potencia Activa (Wh)", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

active_fig_path = os.path.join(SAVE_DIR, "consumo_activo_anual.png")
plt.savefig(active_fig_path, dpi=300)
plt.show()
plt.close()
print(f"Gráfica de consumo activo guardada en: {active_fig_path}")

# 4. GRÁFICO 2: CONSUMO PASIVO (POTENCIA REACTIVA / BASAL - VARh)
print("Generando gráfica de Consumo Pasivo...")
plt.figure(figsize=(16, 6))
plt.plot(df.index, df[reactive_col], color='#ff7f0e', linewidth=1, alpha=0.85)

plt.title("Perfil Anual de Consumo Eléctrico Pasivo / Reactivo (2023)", fontsize=14, fontweight='bold')
plt.xlabel("Fecha", fontsize=11)
plt.ylabel("Potencia Reactiva (VARh)", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

reactive_fig_path = os.path.join(SAVE_DIR, "consumo_pasivo_anual.png")
plt.savefig(reactive_fig_path, dpi=300)
plt.show()
plt.close()
print(f"Gráfica de consumo pasivo guardada en: {reactive_fig_path}")

print("\n¡Visualizaciones listas para el Capítulo 4 (Análisis Exploratorio) de tu TFG!")