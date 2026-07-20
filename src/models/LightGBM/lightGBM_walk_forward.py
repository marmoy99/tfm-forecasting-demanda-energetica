import pandas as pd
from lightgbm import LGBMRegressor
from skforecast.recursive import ForecasterRecursive
import holidays
import os

# Definicion de fichero
CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
FICHERO = os.path.join(CARPETA_SCRIPT, "..", "..", "..", "data", "processed", "dataset_modelado.csv")

# Variables globales, editar a necesidad
REGRESORES_CLIMA = ["HDD", "CDD", "humedad_relativa", "velocidad_viento", "radiacion_solar"]
DIAS_TEST = 3          # horizonte de cada prediccion
STEP_DIAS = 30         # separacion entre cortes (un corte al mes)
N_VENTANAS = 12        # numero de cortes (aprox. un año, las 4 estaciones)
LAGS = 72

# skforecast necesita la fecha como índice y frecuencia horaria fija
df = pd.read_csv(FICHERO, parse_dates=["datetime"]).set_index("datetime").sort_index()

# El estudio termina el 31-mar-2026: abril-2026 es dato provisional de Esios
# (demanda ~40% por debajo de lo real) y quedo excluido por decision del grupo.
df = df[df.index <= "2026-03-31 23:00:00"]
df = df.asfreq("h")
df["demanda_mw"] = df["demanda_mw"].interpolate()   # rellena los huecos del cambio de hora

# LightGBM no entiende de fechas: le damos el calendario en columnas
df["mes"] = df.index.month
df["dia_semana"] = df.index.dayofweek
df["hora"] = df.index.hour
festivos_es = holidays.Spain(years=range(2021, 2028))
df["es_festivo"] = df.index.map(lambda x: x.date() in festivos_es).astype(int)

REGRESORES = REGRESORES_CLIMA + ["mes", "dia_semana", "hora", "es_festivo"]

# Cortes automaticos: desde el final hacia atras, uno cada STEP_DIAS
fin = df.index.max()
cortes = [fin - pd.Timedelta(days=DIAS_TEST + STEP_DIAS * i) for i in range(N_VENTANAS)][::-1]


def metricas(real, estimado):
    mape = (abs(real - estimado) / real).mean() * 100     # error medio en %
    mae = (abs(real - estimado)).mean()                   # error medio en MW
    rmse = (((real - estimado) ** 2).mean()) ** 0.5       # penaliza mas los errores grandes
    return mape, mae, rmse


def evaluar(corte):
    fin_test = pd.Timestamp(corte) + pd.Timedelta(days=DIAS_TEST)
    train = df.loc[df.index < corte]
    test = df.loc[(df.index >= corte) & (df.index < fin_test)]

    # Si el corte no tiene test completo (cerca del final del dataset), se salta
    if len(test) < DIAS_TEST * 24:
        return None

    forecaster = ForecasterRecursive(
        estimator=LGBMRegressor(random_state=123, verbose=-1),
        lags=LAGS,
    )
    forecaster.fit(y=train["demanda_mw"], exog=train[REGRESORES])
    prediccion = forecaster.predict(steps=len(test), exog=test[REGRESORES])
    prediccion.index = test.index

    real = test["demanda_mw"]

    # Real y prediccion deben cubrir exactamente las mismas horas
    if not real.index.equals(prediccion.index):
        raise ValueError(f"Fechas desalineadas en el corte {corte}")

    return metricas(real, prediccion)


mapes, maes, rmses = [], [], []
for corte in cortes:
    resultado = evaluar(corte)
    if resultado is None:
        continue
    mape, mae, rmse = resultado
    mapes.append(mape)
    maes.append(mae)
    rmses.append(rmse)
    print(f"corte {corte:%Y-%m-%d}: MAPE {mape:5.2f}%  MAE {mae:6.0f} MW  RMSE {rmse:6.0f} MW")

n = len(mapes)
print(f"\n=== Media walk-forward LightGBM ({n} ventanas) ===")
print(f"MAPE: {sum(mapes) / n:.2f}%   MAE: {sum(maes) / n:.0f} MW   RMSE: {sum(rmses) / n:.0f} MW")