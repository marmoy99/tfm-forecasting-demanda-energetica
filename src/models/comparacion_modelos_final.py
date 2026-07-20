import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# ---------------------------------------------------------------------------
# Rutas: ajustar según dónde esté este script dentro del repositorio
# ---------------------------------------------------------------------------
CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
RAIZ_REPOSITORIO = os.path.abspath(os.path.join(CARPETA_SCRIPT, "..", ".."))

CARPETA_REPORTS = os.path.join(RAIZ_REPOSITORIO, "reports")

FICHERO_DATASET = os.path.join(
    RAIZ_REPOSITORIO, "data", "processed", "dataset_modelado.csv"
)
FICHERO_PROPHET = os.path.join(CARPETA_REPORTS, "predictions", "prediccion_prophet_final.csv")
FICHERO_SARIMAX = os.path.join(CARPETA_REPORTS, "predictions", "prediccion_sarimax_final.csv")
FICHERO_LIGHTGBM = os.path.join(CARPETA_REPORTS, "predictions", "prediccion_lightgbm_final.csv")

# La gráfica se guarda dentro de reports/figures
FICHERO_GRAFICA = os.path.join(CARPETA_REPORTS, "figures", "comparativa_predicciones_finales.png")
os.makedirs(os.path.dirname(FICHERO_GRAFICA), exist_ok=True)

# Misma fecha de corte que en los tres scripts _final.py, para que todo encaje
FECHA_CORTE_FINAL = "2026-03-23 00:00:00"
DIAS_HISTORICO_GRAFICA = 8
INTERVALO_HORAS_GRAFICA = 12

# ---------------------------------------------------------------------------
# Cargar el histórico real (para dibujar la línea de "Demanda observada")
# ---------------------------------------------------------------------------
df = pd.read_csv(FICHERO_DATASET, parse_dates=["datetime"])
df = df[df["datetime"] <= FECHA_CORTE_FINAL]

fecha_corte = pd.Timestamp(FECHA_CORTE_FINAL)
inicio_historico = fecha_corte - pd.Timedelta(days=DIAS_HISTORICO_GRAFICA)
historico_reciente = df[df["datetime"] >= inicio_historico]

# ---------------------------------------------------------------------------
# Cargar las tres predicciones
# ---------------------------------------------------------------------------
prophet = pd.read_csv(FICHERO_PROPHET, parse_dates=["datetime"])
prophet = prophet.rename(columns={"datetime": "ds", "prediccion_demanda_mw": "yhat"})

sarimax = pd.read_csv(FICHERO_SARIMAX, parse_dates=["datetime"])
sarimax = sarimax.rename(columns={"datetime": "ds", "prediccion_demanda_mw": "yhat"})

lightgbm = pd.read_csv(FICHERO_LIGHTGBM, parse_dates=["datetime"])
lightgbm = lightgbm.rename(columns={"datetime": "ds", "prediccion_demanda_mw": "yhat"})

# ---------------------------------------------------------------------------
# Dibujar: histórico + las tres predicciones superpuestas
# ---------------------------------------------------------------------------
plt.figure(figsize=(14, 5))

plt.plot(historico_reciente["datetime"], historico_reciente["demanda_mw"],
         label="Demanda observada", linewidth=2, color="black")

plt.plot(prophet["ds"], prophet["yhat"], label="Prophet", linewidth=2)
plt.plot(sarimax["ds"], sarimax["yhat"], label="SARIMAX", linewidth=2)
plt.plot(lightgbm["ds"], lightgbm["yhat"], label="LightGBM", linewidth=2)

plt.axvline(fecha_corte, linestyle="--", color="gray", label="Corte (fin de datos)")
plt.legend()
plt.ylabel("MW")
plt.title("Comparativa de predicciones finales: Prophet vs SARIMAX vs LightGBM")

plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=INTERVALO_HORAS_GRAFICA))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))
plt.gcf().autofmt_xdate()

plt.tight_layout()
plt.savefig(FICHERO_GRAFICA, dpi=110)
print(f"Gráfica combinada guardada en: {FICHERO_GRAFICA}")