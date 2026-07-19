import pandas as pd
from prophet import Prophet
import os

# Definicion de ficheros
CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
FICHERO = os.path.join(CARPETA_SCRIPT, "..", "..", "..", "data", "processed", "dataset_modelado.csv")
FICHERO_RESULTADOS = os.path.join(CARPETA_SCRIPT, "..", "..", "..", "reports", "model_results", "resultados_prophet_walk_forward.csv")

# Variables globales, editar a necesidad
REGRESORES = ["HDD", "CDD", "humedad_relativa", "velocidad_viento", "radiacion_solar"]
DIAS_TEST = 3          # horizonte de cada prediccion
STEP_DIAS = 30         # separacion entre cortes (un corte al mes)
N_VENTANAS = 12        # numero de cortes (aprox. un año, las 4 estaciones)

df = pd.read_csv(FICHERO, parse_dates=["datetime"])

# Prophet necesita las columnas 'ds' (fecha) e 'y' (demanda)
d = df[["datetime", "demanda_mw"] + REGRESORES].rename(
    columns={"datetime": "ds", "demanda_mw": "y"}
)

# Cortes automaticos: desde el final hacia atras, uno cada STEP_DIAS
fin = d["ds"].max()
cortes = [fin - pd.Timedelta(days=DIAS_TEST + STEP_DIAS * i) for i in range(N_VENTANAS)][::-1]


def metricas(real, estimado):
    mape = (abs(real - estimado) / real).mean() * 100     # error medio en %
    mae = (abs(real - estimado)).mean()                   # error medio en MW
    rmse = (((real - estimado) ** 2).mean()) ** 0.5       # penaliza mas los errores grandes
    return mape, mae, rmse


def evaluar(corte):
    fin_test = pd.Timestamp(corte) + pd.Timedelta(days=DIAS_TEST)
    train = d[d["ds"] < corte]
    test = d[(d["ds"] >= corte) & (d["ds"] < fin_test)]

    # Si el corte no tiene test completo (cerca del final del dataset), se salta
    if len(test) < DIAS_TEST * 24:
        return None

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

    modelo.fit(train)
    prediccion = modelo.predict(test)

    real = test.set_index("ds")["y"]
    estimado = prediccion.set_index("ds")["yhat"]

    # Real y prediccion deben cubrir exactamente las mismas horas (por el cambio de hora)
    if not real.index.equals(estimado.index):
        raise ValueError(f"Fechas desalineadas en el corte {corte}")

    return metricas(real, estimado)


filas = []
for corte in cortes:
    resultado = evaluar(corte)
    if resultado is None:
        continue
    mape, mae, rmse = resultado
    filas.append({
        "corte": pd.Timestamp(corte).strftime("%Y-%m-%d"),
        "modelo": "Prophet",
        "mape": round(mape, 2),
        "mae": round(mae, 0),
        "rmse": round(rmse, 0),
    })
    print(f"corte {corte:%Y-%m-%d}: MAPE {mape:5.2f}%  MAE {mae:6.0f} MW  RMSE {rmse:6.0f} MW")

resultados = pd.DataFrame(filas)

n = len(resultados)
print(f"\n=== Media walk-forward Prophet ({n} ventanas) ===")
print(f"MAPE: {resultados['mape'].mean():.2f}%   MAE: {resultados['mae'].mean():.0f} MW   RMSE: {resultados['rmse'].mean():.0f} MW")

# Guardar resultados por ventana (para la tabla comparativa entre modelos)
os.makedirs(os.path.dirname(FICHERO_RESULTADOS), exist_ok=True)
resultados.to_csv(FICHERO_RESULTADOS, index=False)
print(f"\nResultados guardados en: {FICHERO_RESULTADOS}")