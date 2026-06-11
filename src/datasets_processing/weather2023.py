import pandas as pd
from meteostat import Point, Hourly
from datetime import datetime
import os

# 1. CONFIGURACIÓN DE LA UBICACIÓN Y FECHAS
cordoba = Point(37.8882, -4.7794, 120)

start = datetime(2023, 1, 1, 0, 0)
end = datetime(2023, 12, 31, 23, 59)

print("Descargando datos horarios de Meteostat para Córdoba (2023)...")

# 2. DESCARGA Y EXTRACCIÓN
data = Hourly(cordoba, start, end)
df_weather = data.fetch()

# 3. LIMPIEZA PARA LA RED NEURONAL
# Nos quedamos con la temperatura ('temp') y la humedad relativa ('rhum')
df_weather = df_weather[['temp', 'rhum']].copy()

# Rellenar posibles huecos (NaN)
nan_temp = df_weather['temp'].isna().sum()
nan_rhum = df_weather['rhum'].isna().sum()

if nan_temp > 0 or nan_rhum > 0:
    print(f"Huecos detectados - Temperatura: {nan_temp}, Humedad: {nan_rhum}. Aplicando interpolación...")
    # La interpolación ahora se aplica a todo el DataFrame (ambas columnas)
    df_weather = df_weather.interpolate()

# Renombramos el índice para que el "join" con tus otros CSVs funcione
df_weather.index.name = "timestamp"

# 4. GUARDADO
output_dir = "../../data/processed"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "dataset_weather2023.csv")

df_weather.to_csv(output_path)

print("\n" + "="*50)
print(f"✓ Dataset guardado en {output_path}")
print(f"Total de filas: {len(df_weather)} (8760)")
print("="*50)
print("\nMuestra de las primeras 5 horas:")
print(df_weather.head())