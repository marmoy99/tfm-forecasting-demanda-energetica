"""
Fase 3b — Validación ROLLING WINDOW (ventana deslizante).  [parte de Oscar]

El compañero valida con EXPANDING window (= TimeSeriesSplit): el train empieza
siempre en 2021 y crece. Aquí probamos la alternativa: ventana de tamaño FIJO
que se desliza. El modelo solo "ve" los últimos N días y olvida lo más antiguo.

Pregunta de investigación (va directa a la memoria):
    ¿Conviene entrenar con TODO el histórico (2021-2026) o solo con lo reciente?
Si la demanda ha cambiado de régimen (post-COVID, electrificación, eficiencia),
el pasado lejano puede estorbar y una ventana corta ganaría.

Comparamos, con el mismo modelo (clima) y las MISMAS ventanas de test:
    - expanding      : train desde 2021 hasta el cutoff (referencia)
    - rolling 365d   : train = último año antes del cutoff
    - rolling 730d   : train = últimos 2 años antes del cutoff
"""
import logging
import warnings

import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

from prophet_baseline import (
    cargar_datos, festivos_espana, metricas, HORIZON, HORIZON_DIAS, FIGS, ROOT,
)

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

STEP_DAYS = 30
N_VENTANAS = 12
REGRESORES = ["HDD", "CDD", "humedad_relativa", "velocidad_viento",
              "radiacion_solar"]
# None = expanding (sin recorte); un nº = días de ventana deslizante
ESTRATEGIAS = {"expanding": None, "rolling 365d": 365, "rolling 730d": 730}


def entrenar_ventana(serie, cutoff, train_days, anios):
    if train_days is None:
        train = serie[serie.ds <= cutoff]
    else:
        ini = cutoff - pd.Timedelta(days=train_days)
        train = serie[(serie.ds > ini) & (serie.ds <= cutoff)]
    test = serie[serie.ds > cutoff].iloc[:HORIZON]
    if len(test) < HORIZON:
        return None

    m = Prophet(holidays=festivos_espana(anios), daily_seasonality=True,
                weekly_seasonality=True, yearly_seasonality=True,
                seasonality_mode="multiplicative")
    for r in REGRESORES:
        m.add_regressor(r)
    m.fit(train)

    futuro = pd.concat([train, test])[["ds", *REGRESORES]]
    pred = m.predict(futuro).iloc[-HORIZON:]["yhat"].to_numpy()
    res = metricas(test["y"].to_numpy(), pred)
    res["n_train"] = len(train)
    return res


def main():
    df = cargar_datos()
    serie = df[["datetime", "demanda_mw", *REGRESORES]].rename(
        columns={"datetime": "ds", "demanda_mw": "y"}
    )
    anios = range(serie.ds.dt.year.min(), serie.ds.dt.year.max() + 2)

    fin = serie.ds.max()
    cutoffs = [fin - pd.Timedelta(days=HORIZON_DIAS + STEP_DAYS * i)
               for i in range(N_VENTANAS)][::-1]

    filas = []
    for nombre, td in ESTRATEGIAS.items():
        for cut in cutoffs:
            r = entrenar_ventana(serie, cut, td, anios)
            if r is None:
                continue
            r.update({"estrategia": nombre, "cutoff": cut})
            filas.append(r)
            print(f"[{nombre:13}] {cut:%Y-%m-%d} -> MAPE {r['MAPE_%']:.2f}% "
                  f"(train {r['n_train']//24}d)")

    r = pd.DataFrame(filas)
    r.to_csv(ROOT / "reports" / "model_results" / "rolling_resultados.csv", index=False)

    print("\n=== EXPANDING vs ROLLING (mismas ventanas, modelo clima) ===")
    resumen = r.groupby("estrategia").agg(
        MAPE_medio=("MAPE_%", "mean"),
        MAPE_std=("MAPE_%", "std"),
        MAPE_max=("MAPE_%", "max"),
    ).round(2).reindex(ESTRATEGIAS.keys())
    print(resumen.to_string())

    fig, ax = plt.subplots(figsize=(13, 5))
    for nombre in ESTRATEGIAS:
        sub = r[r.estrategia == nombre]
        ax.plot(sub["cutoff"], sub["MAPE_%"], marker="o",
                label=f"{nombre} (media {sub['MAPE_%'].mean():.2f}%)")
    ax.axhline(3, color="green", ls="--", lw=1, label="Objetivo TFM (3%)")
    ax.set_title(f"Expanding vs Rolling window — MAPE a {HORIZON_DIAS} días")
    ax.set_ylabel("MAPE (%)")
    ax.set_xlabel("Fecha de corte (cutoff)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGS / "prophet_rolling_vs_expanding.png", dpi=110)
    print(f"\nFigura -> {FIGS / 'prophet_rolling_vs_expanding.png'}")


if __name__ == "__main__":
    main()
