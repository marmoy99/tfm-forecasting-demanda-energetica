import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursive
import holidays
import os

# Definicion de ficheros
CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
FICHERO = os.path.join(CARPETA_SCRIPT, "..", "..", "..", "data", "processed", "dataset_modelado.csv")
FICHERO_GRAFICA = os.path.join(CARPETA_SCRIPT, "..", "..", "..", "reports", "figures", "lightgbm_final.png")
FICHERO_PREDICCION = os.path.join(CARPETA_SCRIPT, "..", "..", "..", "reports", "predictions", "prediccion_lightgbm_final.csv")

# Variables globales, editar a necesidad
REGRESORES_CLIMA = ["HDD", "CDD", "humedad_relativa", "velocidad_viento", "radiacion_solar"]
FECHA_CORTE_FINAL = "2026-03-22 23:00:00"
INTERVALO_HORAS_GRAFICA = 12
DIAS_A_PREDECIR = 3
DIAS_HISTORICO_GRAFICA = 7
LAGS = 72

# skforecast necesita la fecha como índice y frecuencia horaria fija
df = pd.read_csv(FICHERO, parse_dates=["datetime"]).set_index("datetime").sort_index()

# CAMBIO DE LA FECHA: recortamos los datos hasta la fecha de corte fija.
df = df[df.index <= FECHA_CORTE_FINAL]

# Forzamos que el índice termine EXACTAMENTE en la hora de corte
rango_fechas = pd.date_range(
    start=df.index.min(), 
    end=pd.to_datetime(FECHA_CORTE_FINAL), 
    freq="h"
)
df = df.reindex(rango_fechas)

# Interpolamos cualquier hueco (incluidas las horas añadidas al final si faltaban)
df["demanda_mw"] = df["demanda_mw"].interpolate()
df = df.asfreq("h")
df["demanda_mw"] = df["demanda_mw"].interpolate()

festivos_es = holidays.Spain(years=range(2021, 2028))
df["mes"] = df.index.month
df["dia_semana"] = df.index.dayofweek
df["hora"] = df.index.hour
df["es_festivo"] = df.index.map(lambda x: x.date() in festivos_es).astype(int)

REGRESORES = REGRESORES_CLIMA + ["mes", "dia_semana", "hora", "es_festivo"]

# Fechas futuras y sus variables
ultima_fecha = df.index.max()
fechas_futuras = pd.date_range(ultima_fecha + pd.Timedelta(hours=1), periods=DIAS_A_PREDECIR * 24, freq="h")
futuro = pd.DataFrame(index=fechas_futuras)
futuro["mes"] = futuro.index.month
futuro["dia_semana"] = futuro.index.dayofweek
futuro["hora"] = futuro.index.hour
futuro["es_festivo"] = futuro.index.map(lambda x: x.date() in festivos_es).astype(int)

# Clima futuro por perfiles historicos (igual que sarimax y prophet)
historico = df.copy()
historico["mes"] = historico.index.month
historico["hora"] = historico.index.hour
for variable in REGRESORES_CLIMA:
    perfil_mes_hora = historico.groupby(["mes", "hora"])[variable].mean()
    perfil_hora = historico.groupby("hora")[variable].mean()
    media_global = historico[variable].mean()
    valores = []
    for mes, hora in zip(futuro["mes"], futuro["hora"]):
        if (mes, hora) in perfil_mes_hora.index:
            valores.append(perfil_mes_hora.loc[(mes, hora)])
        elif hora in perfil_hora.index:
            valores.append(perfil_hora.loc[hora])
        else:
            valores.append(media_global)
    futuro[variable] = valores

forecaster = ForecasterRecursive(estimator=LGBMRegressor(random_state=123, verbose=-1), lags=LAGS)
forecaster.fit(y=df["demanda_mw"], exog=df[REGRESORES])
prediccion = forecaster.predict(steps=DIAS_A_PREDECIR * 24, exog=futuro[REGRESORES])
print(prediccion.to_string())

# Guardar la predicción para la comparativa entre modelos (columnas ds, yhat)
salida_lgbm = futuro.copy()
salida_lgbm["prediccion_demanda_mw"] = prediccion
salida_lgbm["modelo"] = "LightGBM"
salida_lgbm["fecha_corte_entrenamiento"] = FECHA_CORTE_FINAL
salida_lgbm["lags"] = LAGS
salida_lgbm["meteo_futura_metodo"] = "perfil_historico_medio_mes_hora"

salida_lgbm = salida_lgbm.reset_index().rename(columns={"index": "datetime"})
salida_lgbm.to_csv(FICHERO_PREDICCION, index=False, encoding="utf-8-sig")

# Grafica
inicio_historico = ultima_fecha - pd.Timedelta(days=DIAS_HISTORICO_GRAFICA)
historico_reciente = df.loc[df.index >= inicio_historico, "demanda_mw"]

plt.figure(figsize=(14, 5))
plt.plot(historico_reciente.index, historico_reciente.values, label="Demanda observada", linewidth=2)
plt.plot(prediccion.index, prediccion.values, label="Predicción LightGBM", linewidth=2, color="darkorange")
plt.axvline(ultima_fecha, linestyle="--", color="gray", label="Corte (fin de datos)")
plt.legend()
plt.ylabel("MW")
plt.title(f"Predicción de los próximos {DIAS_A_PREDECIR} días")
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=INTERVALO_HORAS_GRAFICA))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig(FICHERO_GRAFICA, dpi=110)