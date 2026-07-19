# Forecasting de Demanda Energética en España 🔋

**Trabajo Final de Máster — Máster en Data Science e Inteligencia Artificial**  
EBIS Business Techschool · Marzo 2026

---

## Descripción del proyecto

Sistema de predicción de la **demanda eléctrica horaria peninsular** a 3 días (72 h) vista, orientado a optimizar las compras en el mercado mayorista (OMIE) de una comercializadora eléctrica.

**Objetivo cuantificable:** MAPE < 3% en predicción horaria a 3 días, mejorando los baselines estadísticos en al menos un 15% relativo.

---

## Estructura del repositorio

```
tfm-forecasting-demanda-energetica/
├── data/
│   ├── raw/                          # Demanda cruda (usada por el dashboard)
│   ├── processed/
│   │   └── dataset_modelado.csv      # Dataset congelado del proyecto (2021 → abr-2026)
│   └── README_data.txt               # Descripción del dataset y su construcción
├── notebooks/
│   ├── 01_extraccion_demanda_esios.ipynb
│   └── 05_modelos_clasicos.ipynb     # Estudio Prophet / SARIMAX / comparativa
├── src/
│   ├── data/                         # Extracción (Esios, Open-Meteo) y feature engineering
│   └── models/
│       ├── Prophet/                  # Evaluación, walk-forward y predicción final
│       ├── SARIMAX/                  # Clásico, armónico y comparación entre ambos
│       ├── LightGBM/                 # Walk-forward y predicción final
│       ├── prophet_*.py / sarimax_armonico.py   # Estudio walk-forward (comparten prophet_baseline.py)
│       └── comparacion_modelos_final.py         # Figura comparativa de los 3 modelos
├── reports/
│   ├── model_results/                # Resultados walk-forward por modelo
│   ├── model_comparison/             # Comparativas entre modelos
│   ├── predictions/                  # Predicciones finales (72 h)
│   └── figures/                      # Todas las gráficas generadas
├── dashboard.py                      # App Streamlit (+ secciones/ y theme.py)
├── docs/                             # Documentación del proyecto
├── logs/                             # Registro de experimentos y decisiones
├── legacy/                           # Material superado (se conserva por trazabilidad)
├── requirements.txt
└── .env.example                      # Plantilla de credenciales
```

---

## Configuración del entorno

### 1. Clonar el repositorio

```bash
git clone https://github.com/vuestro-usuario/tfm-forecasting-demanda-energetica.git
cd tfm-forecasting-demanda-energetica
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configurar credenciales

```bash
cp .env.example .env
# Editar .env con vuestras API keys reales
```

Necesitáis dos API keys (ambas gratuitas):

| API | Registro | Tiempo |
|-----|----------|--------|
| REE / e·sios | [esios.ree.es](https://esios.ree.es) |
| AEMET OpenData | [opendata.aemet.es](https://opendata.aemet.es) | Inmediato |

### 4. Descargar los datos históricos

```bash
python src/data/extraccion_demanda.py
python src/data/extraccion_meteo.py
python src/data/feature_engineering.py
```

(El dataset ya congelado está versionado en `data/processed/dataset_modelado.csv`; estos scripts solo hacen falta para regenerarlo o ampliarlo.)

---

## Fuentes de datos

| Fuente | Variable | Granularidad | Histórico |
|--------|----------|-------------|-----------|
| REE e·sios (indicador 460) | Demanda real peninsular | Horaria | 2014 – presente |
| AEMET OpenData | Temperatura, humedad, radiación, viento | Diaria/horaria | 2014 – presente |
| OMIE | Precio del pool eléctrico | Horaria | 2014 – presente |

---

## Modelos

| Modelo | Tipo | Rol |
|--------|------|-----|
| Lag 168h | Naive | Baseline mínimo |
| SARIMA | Estadístico | Baseline interpretable |
| Prophet | Estadístico aditivo | Baseline con festivos |
| **LightGBM** | Gradient Boosting | **Modelo principal** |
| LSTM | Deep Learning | Experimento opcional |

---

## Stack tecnológico

Python 3.10+ · pandas · scikit-learn · LightGBM · Prophet · statsmodels · SHAP · Streamlit · Optuna

---

*Documento de planificación completo disponible en `/docs/`*
