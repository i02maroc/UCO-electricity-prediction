# UCO Electricity Prediction

Predicción horaria del consumo eléctrico del edificio Leonardo Da Vinci (Campus de Rabanales, Universidad de Córdoba) mediante redes neuronales recurrentes GRU y LSTM.

Trabajo Fin de Grado — Grado en Ingeniería Informática  
Escuela Politécnica Superior de Córdoba — Universidad de Córdoba  
Autor: Carlos Marín Rodríguez | Director: José Luis Ávila Jiménez


## Estructura del repositorio

- `data/` — Datasets procesados
- `src/` — Scripts de entrenamiento y evaluación por versión y arquitectura
- `src/datasets/` — Scripts de preprocesamiento e integración de datos
- `src/Sarima/` — Implementación del baseline SARIMA
- `src/others/` — Scripts auxiliares de gráficas
- `models/` — Modelos entrenados y escaladores por versión, arquitectura y semilla
- `models/results_*/` — Gráficas y resultados comparativos
- `models/worst_predictions/` — Análisis de los 15 peores errores agregado sobre 20 semillas

---

## Metodología

El trabajo evalúa cinco versiones del modelo (V1 a V5) en paralelo sobre dos arquitecturas recurrentes (GRU y LSTM), siguiendo un protocolo de evaluación estocástica con 20 semillas aleatorias independientes. Las métricas reportadas son el MAPE medio, la desviación típica y los valores extremos sobre las 20 ejecuciones.

| Versión | Variables de entrada | F |
|---|---|---|
| V1 | Codificación cíclica temporal + is_weekend | 8 |
| V2 | V1 + calendario académico UCO | 11 |
| V3 | V2 + temperatura horaria | 12 |
| V4 | V3 + retardos t-24 y t-168 + indicador festivo t+1 | 15 |
| V5 | Igual que V1 + protocolo de entrenamiento refinado | 8 |

---

## Instalación

**1. Clona el repositorio**
```bash
git clone https://github.com/i02maroc/UCO-electricity-prediction
cd UCO-electricity-prediction
```

**2. Crea el entorno conda**
```bash
# Linux / macOS
conda env create -f environment.yml

# Windows
conda env create -f windows_environment.yml
```

**3. Activa el entorno**
```bash
conda activate electricity-prediction
```

---

## Uso

Los scripts de entrenamiento y evaluación se encuentran en `src/vX/` donde X es la versión del modelo (1 a 5). Cada carpeta contiene scripts independientes para GRU y LSTM, así como el script de evaluación con 20 semillas.

Los modelos entrenados, escaladores y resultados de cada semilla se almacenan automáticamente en `models/vX/gru/` y `models/vX/lstm/`.

---

## Tecnologías

- Python 3.10
- TensorFlow / Keras
- scikit-learn
- pandas / numpy / matplotlib
- statsmodels
- Meteostat
- Conda

---

## Licencia

MIT License — véase el archivo [LICENSE](LICENSE) para más detalles.

---

## Referencia

Este repositorio es el manual de código del Trabajo Fin de Grado:

> Carlos Marín Rodríguez, *"Predicción de costes eléctricos de la Universidad de Córdoba mediante redes neuronales"*, Grado en Ingeniería Informática, Escuela Politécnica Superior de Córdoba, Universidad de Córdoba, 2025.
