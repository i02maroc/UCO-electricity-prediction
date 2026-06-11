import pandas as pd
import re
import numpy as np
import os

# PARSER — Lee los CSV raw y los convierte a DataFrame
def parse_csv(file_path: str) -> pd.DataFrame:
    timestamps, values, quality_flags = [], [], []

    with open(file_path, 'r', encoding='utf-8') as f:
        current_day = None
        for line in f:
            line = line.strip()

            # Detectar cabecera de día
            day_match = re.match(r"días(\d{2}/\d{2}/\d{4})", line)
            if day_match:
                current_day = day_match.group(1)
                continue

            # Detectar fila de datos horarios
            row_match = re.match(r"^Horas,(\d+),Consumos:,(-?\d+),Calidad:,(.*)$", line)
            if row_match and current_day:
                hour = int(row_match.group(1))
                if hour == 24:
                    continue  # Hora 24 se encuentra repetida — ignorar

                value = int(row_match.group(2))
                quality_text = row_match.group(3).strip()
                quality = 1 if "Calculada a partir de la curva cuarto-horaria" in quality_text else 0

                ts = pd.to_datetime(f"{current_day} {hour}:00", dayfirst=True)
                timestamps.append(ts)
                values.append(value)
                quality_flags.append(quality)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'consumption': values,
        'quality': quality_flags
    })
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    return df


# CARGA — Parsear archivos raw de energía activa y reactiva
df_active = parse_csv("../../data/raw/Costs2023_Active.csv")
df_reactive = parse_csv("../../data/raw/Costs2023_Reactive.csv")

# Combinar en un único DataFrame alineado por timestamp
df = pd.concat([
    df_active.rename(columns={
        'consumption': 'active_power',
        'quality':     'active_quality'
    }),
    df_reactive.rename(columns={
        'consumption': 'reactive_power',
        'quality':     'reactive_quality'
    })
], axis=1)

# Convertir a float para permitir NaN en pasos posteriores
df['active_power'] = df['active_power'].astype(float)
df['reactive_power'] = df['reactive_power'].astype(float)

# Propagar flags de calidad (ffill+bfill cubre posibles huecos de alineación)
df[['active_quality', 'reactive_quality']] = (
    df[['active_quality', 'reactive_quality']].ffill().bfill().astype(int)
)

# LIMPIEZA — Marcar datos no fiables como NaN antes de interpolar

# Calidad 0 → dato no disponible → NaN
df.loc[df['active_quality'] == 0, 'active_power'] = np.nan

# active_power == 0 con calidad 1 → artefacto del parser original → NaN
# (consumo real de 0 Wh es físicamente imposible en la UCO) - Error en los datos de 24h y 0h
df.loc[(df['active_power'] == 0) & (df['active_quality'] == 1), 'active_power'] = np.nan

print(f"NaN en active_power antes de interpolación: {df['active_power'].isna().sum()}")

# Interpolar huecos <= 3h con método temporal; eliminar huecos más largos
df['active_power'] = df['active_power'].interpolate(method='time', limit=3)
df = df.dropna(subset=['active_power'])

print(f"Filas finales tras limpieza: {len(df)}")

# GUARDADO
df = df[['active_power', 'reactive_power', 'active_quality', 'reactive_quality']]

os.makedirs("../../data/processed", exist_ok=True)
df.to_csv("../../data/processed/dataset_electric2023.csv")

print("\ndataset_electric2023.csv guardado")
print(df.head())