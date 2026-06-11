import pandas as pd
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

# 1. CONFIGURACIÓN DE RUTAS
DATA_PATH = "../../data/processed/dataset_academic2023.csv"
SAVE_DIR = "../../models/results_data_exploration/" 

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# 2. CARGA DE DATOS
print("Cargando dataset académico...")
df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df["fecha_dia"] = df["timestamp"].dt.date

# 3. AGRUPACIÓN POR DÍAS REALES
mapeo_nombres = {
    "is_class": "Período Lectivo",
    "is_admin": "Días Administrativos",
    "is_weekend": "Fines de Semana",
    "is_vacation": "Vacaciones Académicas",
    "is_exam": "Período de Exámenes",
    "is_holiday": "Festivos Oficiales"
}

columnas_validas = [col for col in mapeo_nombres.keys() if col in df.columns]

# Agrupamos por 'fecha_dia' y tomamos el valor máximo
df_diario = df.groupby("fecha_dia")[columnas_validas].max()

# Sumamos los días en los que cada categoría estuvo activa (1)
conteos_dias = df_diario.sum()
conteos_dias = conteos_dias.rename(index=mapeo_nombres)

conteos_dias = conteos_dias.sort_values(ascending=False)

print("\nResumen cuantitativo en DÍAS COMPLETOS (UCO 2023):")
for categoria, dias in conteos_dias.items():
    print(f" - {categoria:<30}: {int(dias):>3} días")

# 4. GENERACIÓN DE LA GRÁFICA
print("\nGenerando gráfica de barras verticales...")
fig, ax = plt.subplots(figsize=(12, 7))

color_palette = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f', '#edc948']

bars = ax.bar(conteos_dias.index, conteos_dias.values, color=color_palette[:len(conteos_dias)], edgecolor='black', width=0.55)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 3, f'{int(height)} días',
            ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2c3e50')

ax.set_title("Distribución de Días Anuales por Categoría Académica (UCO 2023)", fontsize=14, fontweight='bold', pad=20)
ax.set_ylabel("Número Total de Días", fontsize=11, labelpad=10)
ax.set_xlabel("Categoría del Calendario", fontsize=11, labelpad=10)

plt.xticks(rotation=15, ha="right", fontsize=10, fontweight='medium')
ax.set_ylim(0, conteos_dias.max() * 1.12)
ax.grid(True, axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()

# 5. GUARDADO
fig_path = os.path.join(SAVE_DIR, "distribucion_calendario_dias.png")
plt.savefig(fig_path, dpi=300)
plt.close()

print(f"Gráfica vertical de barras por días guardada en: {fig_path}")