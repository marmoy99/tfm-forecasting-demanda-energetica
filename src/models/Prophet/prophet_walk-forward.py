import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import os

# Definicion de fichero
CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
FICHERO = os.path.join(CARPETA_SCRIPT, "..", "Trabajo Miguel", "data", "processed", "dataset_modelado.csv")

# Variables globales, editar a necesidad
REGRESORES = ["HDD", "CDD", "humedad_relativa", "velocidad_viento", "radiacion_solar"]
# Fechas de corte distribuidas por el año (invierno, primavera, verano, otoño)
CORTES = ["2025-02-05", "2025-05-06", "2025-07-08", "2025-11-04"]
# Horizonte de predicción para la evaluación
DIAS_TEST = 3

# Cargar los datos y tratar las fechas como datetime
df = pd.read_csv(FICHERO, parse_dates=["datetime"])

# Prophet necesita las columnas 'ds' (fecha) e 'y' (demanda), se cambia los nombres
d = df[["datetime", "demanda_mw"] + REGRESORES].rename(
    columns={"datetime": "ds", "demanda_mw": "y"}
)

# Realiza una validación temporal (Walk-Forward) para una fecha específica.
def evaluar(corte):

    # Definir el final de la ventana de test según los días configurados
    fin = pd.Timestamp(corte) + pd.Timedelta(days=DIAS_TEST)

    # Dividir el dataset en entrenamiento y prueba
    train = d[d["ds"] < corte]
    test = d[(d["ds"] >= corte) & (d["ds"] < fin)]
 
    # Se configura el modelo
    modelo = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode="multiplicative",
        # Regula la sensibilidad a los cambios de tendencia.
        # Valores más altos = más flexible; valores más bajos = más rígido. (Por defecto: 0.05)
        changepoint_prior_scale=0.05,
    )

    # Añadimos calendario de festivos y los regresores
    modelo.add_country_holidays(country_name="ES")
    for variable in REGRESORES:
        modelo.add_regressor(variable)

    # Entrenamiento del modelo
    modelo.fit(train)

    # Realizar la predicción para el período de test
    prediccion = modelo.predict(test)

    # Alinear los valores reales y predichos mediante el índice temporal para el cálculo del error
    real = test.set_index("ds")["y"]
    estimado = prediccion.set_index("ds")["yhat"]

    # Calcular y retornar el MAPE
    return (abs(real - estimado) / real).mean() * 100

# Evaluar el rendimiento del modelo en cada uno de los cortes temporales
mapes = []
for corte in CORTES:
    mape = evaluar(corte)
    mapes.append(mape)
    print(f"corte {corte}: MAPE {mape:.2f}%")

# Se imprime el MAPE medio de todas las fechas
print(f"\nMAPE medio (walk-forward): {sum(mapes) / len(mapes):.2f}%")