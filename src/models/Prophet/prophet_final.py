import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from prophet import Prophet
import os

# Definicion de ficheros
CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
FICHERO = os.path.join(CARPETA_SCRIPT, "..", "..", "..", "Trabajo Miguel", "data", "processed", "dataset_modelado.csv")
FICHERO_GRAFICA = os.path.join(CARPETA_SCRIPT, "imagenes_Generadas", "prophet_final.png")

# Variables globales, editar a necesidad
REGRESORES = ["HDD", "CDD", "humedad_relativa", "velocidad_viento", "radiacion_solar"]
INTERVALO_HORAS_GRAFICA = 12
DIAS_A_PREDECIR = 3
DIAS_HISTORICO_GRAFICA = 7   # cuántos días reales previos mostrar en el gráfico

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


# Rellenamos el clima: las fechas conocidas con su valor real, las futuras con perfiles
futuro = futuro.merge(d[["ds"] + REGRESORES], on="ds", how="left")
mask_futuro = futuro["ds"] > d["ds"].max()
imputado = imputar_clima_futuro(d, futuro[mask_futuro])
for variable in REGRESORES:
    futuro.loc[mask_futuro, variable] = imputado[variable].values

prediccion = modelo.predict(futuro)

# Solo nos interesan las horas nuevas (las que no estaban en 'd')
nuevas = prediccion[prediccion["ds"] > d["ds"].max()]
print(nuevas[["ds", "yhat"]].to_string(index=False))

# ---------------------------------------------------------------------------
# Gráfica: demanda real de los últimos días + predicción, con línea de corte
# ---------------------------------------------------------------------------
fecha_corte = d["ds"].max()                                  # última hora conocida
inicio_historico = fecha_corte - pd.Timedelta(days=DIAS_HISTORICO_GRAFICA)
historico_reciente = d[d["ds"] >= inicio_historico]          # días reales previos

plt.figure(figsize=(14, 5))
plt.plot(historico_reciente["ds"], historico_reciente["y"],
         label="Demanda observada", linewidth=2)
plt.plot(nuevas["ds"], nuevas["yhat"],
         label="Predicción Prophet", linewidth=2, color="darkorange")
plt.axvline(fecha_corte, linestyle="--", color="gray", label="Corte (fin de datos)")
plt.legend()
plt.ylabel("MW")
plt.title(f"Predicción de los próximos {DIAS_A_PREDECIR} días")

plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=INTERVALO_HORAS_GRAFICA))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
plt.gcf().autofmt_xdate()

plt.tight_layout()
plt.savefig(FICHERO_GRAFICA, dpi=110)