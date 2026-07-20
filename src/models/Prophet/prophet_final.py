import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
import os

# Definicion de ficheros
CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
FICHERO = os.path.join(CARPETA_SCRIPT, "..", "..", "..", "data", "processed", "dataset_modelado.csv")
FICHERO_GRAFICA = os.path.join(CARPETA_SCRIPT, "..", "..", "..", "reports", "figures", "prophet_final.png")
FICHERO_PREDICCION = os.path.join(CARPETA_SCRIPT, "..", "..", "..", "reports", "predictions" ,"prediccion_prophet_final.csv")

# Variables globales, editar a necesidad
REGRESORES = ["HDD", "CDD", "humedad_relativa", "velocidad_viento", "radiacion_solar"]
FECHA_CORTE_FINAL = "2026-03-22 23:00:00"
INTERVALO_HORAS_GRAFICA = 12
DIAS_A_PREDECIR = 3
DIAS_HISTORICO_GRAFICA = 7

df = pd.read_csv(FICHERO, parse_dates=["datetime"])

d = df[["datetime", "demanda_mw"] + REGRESORES].rename(
    columns={"datetime": "ds", "demanda_mw": "y"}
)

# CAMBIO DE LA FECHA: recortamos los datos hasta la fecha de corte fija.
# Así "la última fecha conocida" es siempre la misma, no la que traiga el archivo.
d = d[d["ds"] <= FECHA_CORTE_FINAL]

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

modelo.fit(d)

# Calculamos exactamente hasta cuándo queremos llegar (corte + días a predecir)
fecha_fin_deseada = pd.to_datetime(FECHA_CORTE_FINAL) + pd.Timedelta(days=DIAS_A_PREDECIR)

# Vemos cuál es el último dato real que tiene Prophet
ultima_fecha_real = d["ds"].max()

# Calculamos las horas exactas de diferencia
horas_a_predecir = int((fecha_fin_deseada - ultima_fecha_real).total_seconds() / 3600)

futuro = modelo.make_future_dataframe(periods=horas_a_predecir, freq="h")


def imputar_clima_futuro(historico, futuro):
    """
    Estima el clima de las fechas futuras con perfiles históricos medios.
    Prioridad: media por (mes, hora); si falta, media por hora; si falta, media global.
    Mismo criterio que sarimax_final.py, para que ambos modelos traten igual el clima futuro.
    """
    historico = historico.copy()
    historico["mes"] = historico["ds"].dt.month
    historico["hora"] = historico["ds"].dt.hour

    futuro = futuro.copy()
    futuro["mes"] = futuro["ds"].dt.month
    futuro["hora"] = futuro["ds"].dt.hour

    for variable in REGRESORES:
        perfil_mes_hora = historico.groupby(["mes", "hora"])[variable].mean()
        perfil_hora = historico.groupby("hora")[variable].mean()
        media_global = historico[variable].mean()

        valores = []
        for _, fila in futuro.iterrows():
            clave = (fila["mes"], fila["hora"])
            if clave in perfil_mes_hora.index:
                valores.append(perfil_mes_hora.loc[clave])
            elif fila["hora"] in perfil_hora.index:
                valores.append(perfil_hora.loc[fila["hora"]])
            else:
                valores.append(media_global)
        futuro[variable] = valores

    return futuro


futuro = futuro.merge(d[["ds"] + REGRESORES], on="ds", how="left")
mask_futuro = futuro["ds"] > d["ds"].max()
imputado = imputar_clima_futuro(d, futuro[mask_futuro])
for variable in REGRESORES:
    futuro.loc[mask_futuro, variable] = imputado[variable].values

prediccion = modelo.predict(futuro)

nuevas = prediccion[prediccion["ds"] >= "2026-03-23 00:00:00"]
print(nuevas[["ds", "yhat"]].to_string(index=False))

# Guardar la predicción para la comparativa entre modelos
salida_prophet = futuro[futuro["ds"] >= "2026-03-23 00:00:00"].copy()

# Le pegamos la predicción calculada ('yhat')
salida_prophet = salida_prophet.merge(nuevas[["ds", "yhat"]], on="ds", how="left")

# Renombramos para usar la misma nomenclatura que SARIMAX y LightGBM
salida_prophet = salida_prophet.rename(columns={"ds": "datetime", "yhat": "prediccion_demanda_mw"})

# Añadimos la trazabilidad
salida_prophet["modelo"] = "Prophet"
salida_prophet["fecha_corte_entrenamiento"] = FECHA_CORTE_FINAL
salida_prophet["meteo_futura_metodo"] = "perfil_historico_medio_mes_hora"

salida_prophet.to_csv(FICHERO_PREDICCION, index=False, encoding="utf-8-sig")

# Grafica: historico reciente + prediccion, con linea de corte
fecha_corte = d["ds"].max()
inicio_historico = fecha_corte - pd.Timedelta(days=DIAS_HISTORICO_GRAFICA)
historico_reciente = d[d["ds"] >= inicio_historico]

plt.figure(figsize=(14, 5))
plt.plot(historico_reciente["ds"], historico_reciente["y"], label="Demanda observada", linewidth=2)
plt.plot(nuevas["ds"], nuevas["yhat"], label="Predicción Prophet", linewidth=2, color="darkorange")
plt.axvline(fecha_corte, linestyle="--", color="gray", label="Corte (fin de datos)")
plt.legend()
plt.ylabel("MW")
plt.title(f"Predicción de los próximos {DIAS_A_PREDECIR} días")
plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=INTERVALO_HORAS_GRAFICA))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig(FICHERO_GRAFICA, dpi=110)
