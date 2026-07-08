import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
import os

# Definicion de ficheros
CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
FICHERO = os.path.join(CARPETA_SCRIPT, "..", "Trabajo Miguel", "data", "processed", "dataset_modelado.csv")
FICHERO_GRAFICA = os.path.join(CARPETA_SCRIPT, "imagenes_Generadas", "prophet_final.png")

# Variables globales, editar a necesidad
REGRESORES = ["HDD", "CDD", "humedad_relativa", "velocidad_viento", "radiacion_solar"]
INTERVALO_HORAS_GRAFICA = 6
DIAS_A_PREDECIR = 3

df = pd.read_csv(FICHERO, parse_dates=["datetime"])

d = df[["datetime", "demanda_mw"] + REGRESORES].rename(
    columns={"datetime": "ds", "demanda_mw": "y"}
)

modelo = Prophet(
    daily_seasonality=True,
    weekly_seasonality=True,
    yearly_seasonality=True,
    seasonality_mode="multiplicative",
    changepoint_prior_scale=0.01,
)
modelo.add_country_holidays(country_name="ES")
for variable in REGRESORES:
    modelo.add_regressor(variable)

# Entrenamos con todo el histórico (ya no reservamos test: aquí no evaluamos, predecimos)
modelo.fit(d)

# Generamos las fechas futuras (las próximas DIAS_A_PREDECIR, hora a hora)
futuro = modelo.make_future_dataframe(periods=DIAS_A_PREDECIR * 24, freq="h")

# El clima futuro no existe en nuestros datos. Aproximación: repetir el último
# valor conocido de cada regresor en todas las horas futuras.
# (En producción real, aquí iría la previsión meteorológica de Open-Meteo/AEMET)
futuro = futuro.merge(d[["ds"] + REGRESORES], on="ds", how="left")
ultimo_valor = d[REGRESORES].iloc[-1]
for variable in REGRESORES:
    futuro[variable] = futuro[variable].fillna(ultimo_valor[variable])

prediccion = modelo.predict(futuro)

# Solo nos interesan las horas nuevas (las que no estaban en 'd')
nuevas = prediccion[prediccion["ds"] > d["ds"].max()]
print(nuevas[["ds", "yhat"]].to_string(index=False))

# Pintar grafica comparativa
plt.figure(figsize=(14, 5))
plt.plot(nuevas.set_index("ds")["yhat"], label="Predicción Prophet", linewidth=2, color="darkorange")
plt.legend()
plt.ylabel("MW")
plt.title(f"Predicción de los próximos {DIAS_A_PREDECIR} días")

# Eje vertical: ya no hay 'real', solo la predicción
estimado = nuevas.set_index("ds")["yhat"]
limite_inferior = int(estimado.min() // 1000) * 1000
limite_superior = int(estimado.max() // 1000) * 1000 + 1000

plt.ylim(limite_inferior, limite_superior)
plt.yticks(np.arange(limite_inferior, limite_superior + 1000, 1000))

# Eje horizontal: igual que en el otro script
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=INTERVALO_HORAS_GRAFICA))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%a %d/%m %H:%M"))
plt.gcf().autofmt_xdate()

plt.tight_layout()
plt.savefig(FICHERO_GRAFICA, dpi=110)