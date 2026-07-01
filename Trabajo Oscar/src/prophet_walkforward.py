"""
Fase 3 — Validación WALK-FORWARD de Prophet.

Un solo holdout (1 semana) no es fiable: esa semana pudo ser fácil o difícil.
El walk-forward evalúa el modelo en MUCHAS semanas, recorriendo el calendario:

    cutoff_1 -> entreno hasta ahí, predigo los 7 días siguientes, mido MAPE
    cutoff_2 = cutoff_1 + 30 días -> idem
    ...

Así obtenemos un MAPE MEDIO y su DESVIACIÓN sobre distintas épocas del año
(invierno, primavera, verano, otoño). Ese es el número honesto para el TFM.

Comparamos dos modelos en cada ventana:
  - baseline : calendario + festivos
  - clima    : + HDD, CDD, humedad, viento, radiación
para demostrar que el clima mejora de forma ROBUSTA, no solo en una semana.
"""
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

from prophet_baseline import (
    cargar_datos, festivos_espana, metricas, HORIZON, FIGS, ROOT,
)

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

STEP_DAYS = 30          # separación entre cutoffs
N_VENTANAS = 12         # ~1 año de validación (las 4 estaciones)
MODELOS = {
    "baseline": [],
    "clima":    ["HDD", "CDD", "humedad_relativa",
                 "velocidad_viento", "radiacion_solar"],
}


def entrenar_ventana(serie, cutoff, regresores, anios):
    train = serie[serie.ds <= cutoff]
    test = serie[(serie.ds > cutoff)].iloc[:HORIZON]
    if len(test) < HORIZON:
        return None

    m = Prophet(
        holidays=festivos_espana(anios),
        daily_seasonality=True, weekly_seasonality=True,
        yearly_seasonality=True, seasonality_mode="multiplicative",
    )
    for r in regresores:
        m.add_regressor(r)
    m.fit(train)

    futuro = pd.concat([train, test])[["ds", *regresores]]
    fcst = m.predict(futuro)
    pred = fcst.iloc[-HORIZON:]["yhat"].to_numpy()
    return metricas(test["y"].to_numpy(), pred)


def main():
    df = cargar_datos()
    cols = sorted({c for regs in MODELOS.values() for c in regs})
    serie = df[["datetime", "demanda_mw", *cols]].rename(
        columns={"datetime": "ds", "demanda_mw": "y"}
    )
    anios = range(serie.ds.dt.year.min(), serie.ds.dt.year.max() + 2)

    # Cutoffs: hacia atrás desde (fin - 7 días), cada STEP_DAYS
    fin = serie.ds.max()
    cutoffs = [fin - pd.Timedelta(days=7 + STEP_DAYS * i)
               for i in range(N_VENTANAS)][::-1]

    filas = []
    for nombre, regs in MODELOS.items():
        for cut in cutoffs:
            res = entrenar_ventana(serie, cut, regs, anios)
            if res is None:
                continue
            res.update({"modelo": nombre, "cutoff": cut})
            filas.append(res)
            print(f"[{nombre:8}] cutoff {cut:%Y-%m-%d} -> "
                  f"MAPE {res['MAPE_%']:.2f}%")

    r = pd.DataFrame(filas)
    out_csv = ROOT / "reports" / "walkforward_resultados.csv"
    r.to_csv(out_csv, index=False)

    print("\n=== RESUMEN WALK-FORWARD ({} ventanas/modelo) ===".format(
        r.modelo.value_counts().min()))
    resumen = r.groupby("modelo").agg(
        MAPE_medio=("MAPE_%", "mean"),
        MAPE_std=("MAPE_%", "std"),
        MAPE_max=("MAPE_%", "max"),
        MAE_medio=("MAE_MW", "mean"),
    ).round(2)
    print(resumen.to_string())
    print(f"\nCSV -> {out_csv}")

    # Figura: MAPE por ventana (línea temporal) + media
    fig, ax = plt.subplots(figsize=(13, 5))
    for nombre in MODELOS:
        sub = r[r.modelo == nombre]
        ax.plot(sub["cutoff"], sub["MAPE_%"], marker="o",
                label=f"{nombre} (media {sub['MAPE_%'].mean():.2f}%)")
    ax.axhline(3, color="green", ls="--", lw=1, label="Objetivo TFM (3%)")
    ax.set_title("Walk-forward: MAPE a 7 días por ventana")
    ax.set_ylabel("MAPE (%)")
    ax.set_xlabel("Fecha de corte (cutoff)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGS / "prophet_walkforward.png", dpi=110)
    print(f"Figura -> {FIGS / 'prophet_walkforward.png'}")


if __name__ == "__main__":
    main()
