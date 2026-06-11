import pandas as pd
import numpy as np
import os

# Fuente: Calendario Académico UCO 2022-2023 (BOUCO 28/12/2022)
# Fuente: Calendario Académico UCO 2023-2024 (CGO 22/12/2022)

idx = pd.date_range("2023-01-01", "2023-12-31 23:00", freq="h")
df = pd.DataFrame(index=idx)
df.index.name = "timestamp"
fecha = df.index.normalize()

# 1. FIN DE SEMANA
df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)

# 2. FESTIVOS OFICIALES
festivos = pd.to_datetime([
    # --- Curso 2022-2023 (enero - agosto) ---
    "2023-01-02",  # Por traslado de Año Nuevo
    "2023-01-06",  # Epifanía del Señor
    "2023-01-27",  # Por traslado de Santo Tomás de Aquino
    "2023-02-28",  # Día de Andalucía
    "2023-03-20",  # San José — Fiesta Patronal EPSC (traslado 19 mar)
    "2023-04-06",  # Jueves Santo
    "2023-04-07",  # Viernes Santo
    "2023-05-01",  # Fiesta del Trabajo
    "2023-05-25",  # Feria Ntra. Sra. de la Salud — jueves
    "2023-05-26",  # Feria Ntra. Sra. de la Salud — viernes
    "2023-08-15",  # Festividad de la Asunción de la Virgen

    # --- Curso 2023-2024 (septiembre - diciembre) ---
    "2023-09-08",  # Ntra. Sra. de la Fuensanta (Córdoba)
    "2023-09-29",  # Acto Oficial Apertura de Curso (no lectivo y no laborable)
    "2023-10-12",  # Fiesta Nacional de España
    "2023-10-24",  # San Rafael (festivo Córdoba capital, salvo EPSB y FISIDEC)
    "2023-11-01",  # Día de Todos los Santos
    "2023-11-17",  # San Alberto Magno — Fiesta Patronal Fac. Ciencias (no afecta a EPSC pero sí cierra parte del campus)
    "2023-12-06",  # Día de la Constitución Española
    "2023-12-08",  # Inmaculada Concepción
    "2023-12-25",  # Natividad del Señor
])

df["is_holiday"] = fecha.isin(festivos).astype(int)

# 3. PERIODOS VACACIONALES / NO LECTIVOS
# Navidad curso 2022-23: 23 dic 2022 → 6 ene 2023 → en dataset: 1-6 enero
vac_navidad_2223 = (fecha >= "2023-01-01") & (fecha <= "2023-01-06")
# Semana Santa: 31 mar → 9 abril 2023
vac_semana_santa = (fecha >= "2023-03-31") & (fecha <= "2023-04-09")
# Verano post-exámenes: julio y agosto (sin actividad lectiva)
vac_verano = (fecha >= "2023-07-09") & (fecha <= "2023-08-31")
# Navidad curso 2023-24: 23 dic → 7 ene 2024 → en dataset: 23-31 dic
vac_navidad_2324 = (fecha >= "2023-12-23") & (fecha <= "2023-12-31")

df["is_vacation"] = (
    vac_navidad_2223 | vac_semana_santa | vac_verano | vac_navidad_2324
).astype(int)

# =============================================================================
# 4. TIPO DE DÍA ACADÉMICO
#
#   0 = No laborable (festivo / fin de semana / vacaciones)
#   1 = Periodo lectivo (clases activas)
#   2 = Periodo de exámenes
#   3 = Periodo administrativo (sin clases, actividad reducida)
# =============================================================================
dia = pd.Series(0, index=fecha, dtype=int)

# -- CURSO 2022-2023 --
# Exámenes 1ª conv. 1er cuatrimestre: 9 ene → 24 ene
dia[(fecha >= "2023-01-09") & (fecha <= "2023-01-24")] = 2
# Exámenes 2ª conv. 1er cuatrimestre: 30 ene → 11 feb
dia[(fecha >= "2023-01-30") & (fecha <= "2023-02-11")] = 2
# Clases 2º cuatrimestre: 13 feb → 24 may
dia[(fecha >= "2023-02-13") & (fecha <= "2023-05-24")] = 1
# Exámenes 1ª conv. 2º cuatrimestre: 29 may → 17 jun
dia[(fecha >= "2023-05-29") & (fecha <= "2023-06-17")] = 2
# Exámenes 2ª conv. 2º cuatrimestre: 26 jun → 8 jul
dia[(fecha >= "2023-06-26") & (fecha <= "2023-07-08")] = 2
# Conv. extraordinaria septiembre: 1 sep → 7 sep
dia[(fecha >= "2023-09-01") & (fecha <= "2023-09-07")] = 2

# -- CURSO 2023-2024 --
# Periodo administrativo pre-clases: 8 sep → 17 sep
dia[(fecha >= "2023-09-08") & (fecha <= "2023-09-17")] = 3
# Clases 1er cuatrimestre: 18 sep → 22 dic (antes de navidad)
dia[(fecha >= "2023-09-18") & (fecha <= "2023-12-22")] = 1

# Sobreescribir con 0 todo lo no laborable
es_no_laborable = (
    fecha.isin(festivos)        |
    (df.index.dayofweek >= 5)   |
    vac_navidad_2223            |
    vac_semana_santa            |
    vac_verano                  |
    vac_navidad_2324
)
dia[es_no_laborable] = 0

df["day_type"] = dia.values

# 5. ONE-HOT ENCODING del day_type (correcto para redes neuronales)
df["is_class"] = (df["day_type"] == 1).astype(int)
df["is_exam"] = (df["day_type"] == 2).astype(int)
df["is_admin"] = (df["day_type"] == 3).astype(int)
# day_type==0 queda implícito cuando las tres anteriores son 0

# 6. VERIFICAR DATOS
print("=== Distribución day_type (días únicos) ===")
resumen = df["day_type"].resample("D").first()
print(resumen.value_counts().sort_index())
print(f"\n  0=No laborable:   {(df['day_type']==0).sum()} horas ({(resumen==0).sum()} días)")
print(f"  1=Lectivo:        {(df['day_type']==1).sum()} horas ({(resumen==1).sum()} días)")
print(f"  2=Exámenes:       {(df['day_type']==2).sum()} horas ({(resumen==2).sum()} días)")
print(f"  3=Administrativo: {(df['day_type']==3).sum()} horas ({(resumen==3).sum()} días)")

print("\n=== Festivos 2023 ===")
for f in sorted(festivos):
    print(f"  {f.strftime('%d/%m/%Y')}  {f.strftime('%A')}")

print("\n=== Septiembre-Diciembre 2023 (muestra diaria) ===")
print(df["2023-09":"2023-12"]["day_type"].resample("D").first().to_string())

# 7. GUARDAR EL DATASET
os.makedirs("../../data/processed", exist_ok=True)
df.to_csv("../../data/processed/dataset_academic2023.csv")
print(f"\ndataset_calendario.csv guardado")
print(f"  Filas: {len(df)} | Columnas: {list(df.columns)}")