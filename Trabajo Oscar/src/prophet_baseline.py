"""
Fase 1 — Baseline Prophet para forecasting de demanda eléctrica horaria.

Objetivo de esta fase: tener un PRIMER número honesto con el modelo más simple
posible (solo estacionalidades + festivos, SIN clima). Sirve de referencia:
todo lo que añadamos después (clima, tuning) tiene que MEJORAR este MAPE.

Estrategia: holdout de los últimos 7 días del dataset.
  - Entrenamos con todo el histórico menos la última semana.
  - Predecimos esos 7 días (168 horas) -> es justo el horizonte del caso de negocio.
  - Medimos MAPE / MAE / RMSE comparando con la demanda real.
"""
from pathlib import Path
import warnings
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import holidays
from prophet import Prophet

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "dataset_modelado.csv"
FIGS = ROOT / "reports" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

HORIZON = 24 * 7  # 168 horas = 7 días
DEMANDA_MIN_VALIDA = 10_000  # MW; por debajo = dato corrupto -> se interpola
# Abril 2026 viene corrupto en la fuente (cae ~40% el 1-abr de golpe, dato
# provisional/incompleto de Esios). Cortamos hasta el último día fiable.
FECHA_MAX = "2026-03-31 23:00:00"
# Apagón ibérico del 28-abr-2025: demanda real cae a ~2.000 MW y tarda ~36h en
# normalizarse. Evento excepcional NO representativo -> lo imputamos por
# "semana análoga" (media de la misma hora 7 días antes y 7 días después),
# que conserva el patrón día/noche. Documentado como imputación de evento.
APAGON_INI = "2025-04-28 12:00:00"
APAGON_FIN = "2025-04-29 23:00:00"


# ----------------------------------------------------------------------------
# 1) Carga y limpieza
# ----------------------------------------------------------------------------
def cargar_datos() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["datetime"]).sort_values("datetime")

    # Corte por datos corruptos al final de la serie (ver FECHA_MAX)
    df = df[df["datetime"] <= FECHA_MAX].copy()

    # Outliers imposibles (p.ej. todo el 2025-10-13 ~1500 MW) -> NaN -> interpolar
    n_malos = (df["demanda_mw"] < DEMANDA_MIN_VALIDA).sum()
    df.loc[df["demanda_mw"] < DEMANDA_MIN_VALIDA, "demanda_mw"] = np.nan
    df["demanda_mw"] = (
        df.set_index("datetime")["demanda_mw"]
        .interpolate(method="time")
        .to_numpy()
    )
    print(f"Limpieza: {n_malos} valores < {DEMANDA_MIN_VALIDA} MW interpolados.")

    # Imputación del apagón ibérico por semana análoga (t-7d y t+7d)
    s = df.set_index("datetime")["demanda_mw"]
    ventana = s.loc[APAGON_INI:APAGON_FIN].index
    prev = s.reindex(ventana - pd.Timedelta(days=7)).to_numpy()
    post = s.reindex(ventana + pd.Timedelta(days=7)).to_numpy()
    s.loc[ventana] = np.nanmean([prev, post], axis=0)
    df["demanda_mw"] = s.to_numpy()
    print(f"Apagón 28-29 abr 2025: {len(ventana)} horas imputadas (semana análoga).")
    return df


def festivos_espana(anios) -> pd.DataFrame:
    """DataFrame de festivos nacionales en el formato que espera Prophet."""
    es = holidays.Spain(years=anios)
    return pd.DataFrame(
        {"holiday": "festivo_es", "ds": pd.to_datetime(list(es.keys()))}
    )


# ----------------------------------------------------------------------------
# 2) Métricas
# ----------------------------------------------------------------------------
def metricas(y_real, y_pred) -> dict:
    y_real, y_pred = np.asarray(y_real), np.asarray(y_pred)
    mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100
    mae = np.mean(np.abs(y_real - y_pred))
    rmse = np.sqrt(np.mean((y_real - y_pred) ** 2))
    return {"MAPE_%": mape, "MAE_MW": mae, "RMSE_MW": rmse}


# ----------------------------------------------------------------------------
# 3) Baseline Prophet
# ----------------------------------------------------------------------------
def main():
    df = cargar_datos()

    # Prophet exige columnas 'ds' (fecha) e 'y' (target)
    serie = df[["datetime", "demanda_mw"]].rename(
        columns={"datetime": "ds", "demanda_mw": "y"}
    )

    # Split: últimas 168h = test, el resto = train
    train = serie.iloc[:-HORIZON].copy()
    test = serie.iloc[-HORIZON:].copy()
    print(f"Train: {train.ds.min()} -> {train.ds.max()}  ({len(train)} h)")
    print(f"Test : {test.ds.min()} -> {test.ds.max()}  ({len(test)} h)")

    anios = range(serie.ds.dt.year.min(), serie.ds.dt.year.max() + 2)
    modelo = Prophet(
        holidays=festivos_espana(anios),
        daily_seasonality=True,    # patrón intradía (clave en demanda)
        weekly_seasonality=True,   # laborable vs finde
        yearly_seasonality=True,   # invierno/verano
        seasonality_mode="multiplicative",  # la amplitud crece con el nivel
    )
    modelo.fit(train)

    # Predicción de las 168h de test
    futuro = modelo.make_future_dataframe(periods=HORIZON, freq="h")
    fcst = modelo.predict(futuro)
    pred = fcst.iloc[-HORIZON:][["ds", "yhat", "yhat_lower", "yhat_upper"]]

    m = metricas(test["y"].to_numpy(), pred["yhat"].to_numpy())
    print("\n=== BASELINE PROPHET (holdout 7 días) ===")
    for k, v in m.items():
        print(f"  {k:10}: {v:8.2f}")

    # ---- Figura 1: real vs predicho en la semana de test ----
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(test["ds"], test["y"], label="Real", color="#222", lw=1.8)
    ax.plot(pred["ds"], pred["yhat"], label="Prophet", color="#e25", lw=1.8)
    ax.fill_between(pred["ds"], pred["yhat_lower"], pred["yhat_upper"],
                    color="#e25", alpha=0.15, label="Intervalo 80%")
    ax.set_title(f"Baseline Prophet — MAPE {m['MAPE_%']:.2f}%  (7 días)")
    ax.set_ylabel("Demanda (MW)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGS / "prophet_baseline_holdout.png", dpi=110)
    print(f"\nFigura -> {FIGS / 'prophet_baseline_holdout.png'}")

    # ---- Figura 2: descomposición (lo interpretable de Prophet) ----
    fig2 = modelo.plot_components(fcst)
    fig2.savefig(FIGS / "prophet_baseline_componentes.png", dpi=110)
    print(f"Figura -> {FIGS / 'prophet_baseline_componentes.png'}")


if __name__ == "__main__":
    main()
