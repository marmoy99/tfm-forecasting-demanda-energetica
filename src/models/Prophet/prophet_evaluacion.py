import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
import os


# Definicion de ficheros
CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
FICHERO = os.path.join(CARPETA_SCRIPT, "..", "..", "..", "Trabajo Miguel", "data", "processed", "dataset_modelado.csv")
FICHERO_GRAFICA = os.path.join(CARPETA_SCRIPT, "imagenes_Generadas", "prophet_evaluacion.png")

# Variables globales, editar a necesidad
REGRESORES = ["HDD", "CDD", "humedad_relativa", "velocidad_viento", "radiacion_solar"]
FECHA_CORTE = "2025-10-03"
INTERVALO_HORAS_GRAFICA = 6
DIAS_TEST = 3

# Cargar los datos y tratar las fechas como datetime
df = pd.read_csv(FICHERO, parse_dates=["datetime"])

# Prophet necesita las columnas 'ds' (fecha) e 'y' (demanda)
d = df[["datetime", "demanda_mw"] + REGRESORES].rename(
    columns={"datetime": "ds", "demanda_mw": "y"}
)

# Definir el final de la ventana de test según la fecha de corte
fin = pd.Timestamp(FECHA_CORTE) + pd.Timedelta(days=DIAS_TEST)
# Dividir el dataset en entrenamiento y prueba
train = d[d["ds"] < FECHA_CORTE]
test = d[(d["ds"] >= FECHA_CORTE) & (d["ds"] < fin)]

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

# Pintar grafica comparativa
plt.figure(figsize=(14, 5))
plt.plot(real, label="Demanda real", linewidth=2)
plt.plot(estimado, label="Predicción Prophet", linewidth=2)
plt.legend()
plt.ylabel("MW")
plt.title(f"Corte {FECHA_CORTE} + {DIAS_TEST} días")

minimo = min(real.min(), estimado.min())
maximo = max(real.max(), estimado.max())
limite_inferior = int(minimo // 1000) * 1000
limite_superior = int(maximo // 1000) * 1000 + 1000

plt.ylim(limite_inferior, limite_superior)
plt.yticks(np.arange(limite_inferior, limite_superior + 1000, 1000))

plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=INTERVALO_HORAS_GRAFICA))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%a %d/%m %H:%M"))
plt.gcf().autofmt_xdate()

plt.tight_layout()
plt.savefig(FICHERO_GRAFICA, dpi=110)

# Calcular el MAPE
mape = (abs(real - estimado) / real).mean() * 100

# Se imprime el MAPE medio de todas las fechas
print(f"MAPE (corte {FECHA_CORTE}, {DIAS_TEST} días): {mape:.2f}%")