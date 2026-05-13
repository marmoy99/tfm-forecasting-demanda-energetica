# Forecasting de Demanda Energética en España 🔋

**Trabajo Final de Máster — Máster en Data Science e Inteligencia Artificial**  
EBIS Business Techschool · Marzo 2026

---

## Descripción del proyecto

Sistema de predicción de la **demanda eléctrica horaria peninsular** a 7 días vista, orientado a optimizar las compras en el mercado mayorista (OMIE) de una comercializadora eléctrica.

**Objetivo cuantificable:** MAPE < 3% en predicción horaria a 7 días, mejorando los baselines estadísticos en al menos un 15% relativo.

---

## Estructura del repositorio

```
tfm-forecasting-demanda-energetica/
├── data/
│   ├── raw/                  # Datos descargados sin modificar (no versionados)
│   └── processed/            # Dataset limpio y features (no versionados)
├── notebooks/                # Jupyter Notebooks de EDA y experimentación
├── src/
│   ├── ingestion/            # Scripts de descarga (REE, AEMET)
│   ├── preprocessing/        # Limpieza y feature engineering
│   ├── models/               # Entrenamiento e inferencia
│   └── evaluation/           # Métricas y comparativas
├── models/                   # Modelos serializados (no versionados)
├── dashboard/                # App Streamlit
├── docs/                     # Memoria y presentación
├── config.py                 # Constantes y configuración global
├── requirements.txt
└── .env.example              # Plantilla de credenciales
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
| REE / e·sios | [esios.ree.es](https://esios.ree.es) | 1-2 días laborables |
| AEMET OpenData | [opendata.aemet.es](https://opendata.aemet.es) | Inmediato |

### 4. Descargar los datos históricos

```bash
python src/ingestion/fetch_ree.py
python src/ingestion/fetch_aemet.py
python src/ingestion/build_dataset.py
```

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

## Equipo

| Persona | Rol |
|---------|-----|
| A | Data Engineer |
| B | Analista / EDA |
| C | ML Engineer |
| D | Business + Deploy |

---

## Stack tecnológico

Python 3.10+ · pandas · scikit-learn · LightGBM · Prophet · statsmodels · SHAP · Streamlit · Optuna

---

*Documento de planificación completo disponible en `/docs/`*
