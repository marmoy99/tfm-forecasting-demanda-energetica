"""
Fase 2 — Prophet + clima (regresores externos).

Partimos del baseline (solo calendario + festivos, MAPE ~7%) y añadimos
variables climáticas como regresores con add_regressor(). La temperatura mueve
mucho la demanda (calefacción/refrigeración), así que esperamos bajar el error.

Probamos varias combinaciones para VER cuánto aporta cada bloque:
  - baseline      : sin clima (referencia)
  - HDD+CDD       : el bloque de temperatura (la forma de U)
  - +meteo        : añade humedad, viento y radiación solar

OJO leakage: un regresor necesita su valor también en el futuro. Aquí usamos el
clima REAL observado en el test = asumir "previsión meteo perfecta". Es lo
estándar para aislar el valor del modelo; en producción vendría de una
previsión a 7 días (AEMET/Open-Meteo), que a ese horizonte es muy buena.
"""
from prophet import Prophet

# Reutilizamos la lógica ya escrita y validada del baseline
from prophet_baseline import (
    cargar_datos, festivos_espana, metricas, HORIZON, FIGS,
)
import matplotlib.pyplot as plt
import pandas as pd

# Combinaciones de regresores a comparar
EXPERIMENTOS = {
    "baseline (sin clima)": [],
    "HDD + CDD":            ["HDD", "CDD"],
    "HDD+CDD + meteo":      ["HDD", "CDD", "humedad_relativa",
                              "velocidad_viento", "radiacion_solar"],
}


def entrenar(serie: pd.DataFrame, regresores: list[str], anios) -> dict:
    train = serie.iloc[:-HORIZON].copy()
    test = serie.iloc[-HORIZON:].copy()

    modelo = Prophet(
        holidays=festivos_espana(anios),
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode="multiplicative",
    )
    for r in regresores:
        modelo.add_regressor(r)
    modelo.fit(train)

    # El futuro debe llevar las columnas de los regresores (valores reales test)
    futuro = serie[["ds", *regresores]].copy()
    fcst = modelo.predict(futuro)
    pred = fcst.iloc[-HORIZON:]

    m = metricas(test["y"].to_numpy(), pred["yhat"].to_numpy())
    return {"pred": pred, "test": test, **m}


def main():
    df = cargar_datos()  # ya viene limpio y cortado en 2026-03-31
    cols_reg = sorted({c for regs in EXPERIMENTOS.values() for c in regs})
    serie = df[["datetime", "demanda_mw", *cols_reg]].rename(
        columns={"datetime": "ds", "demanda_mw": "y"}
    )
    anios = range(serie.ds.dt.year.min(), serie.ds.dt.year.max() + 2)

    resultados = {}
    for nombre, regs in EXPERIMENTOS.items():
        print(f"Entrenando: {nombre} ...")
        resultados[nombre] = entrenar(serie, regs, anios)

    print("\n=== COMPARATIVA (holdout 7 días, 25-31 mar 2026) ===")
    print(f"{'Modelo':22} {'MAPE %':>8} {'MAE MW':>9} {'RMSE MW':>9}")
    base = resultados["baseline (sin clima)"]["MAPE_%"]
    for nombre, r in resultados.items():
        mejora = "" if nombre.startswith("baseline") else \
            f"  ({(base - r['MAPE_%']) / base * 100:+.0f}% vs baseline)"
        print(f"{nombre:22} {r['MAPE_%']:8.2f} {r['MAE_MW']:9.0f} "
              f"{r['RMSE_MW']:9.0f}{mejora}")

    # Figura: real vs cada modelo
    mejor = min(resultados, key=lambda k: resultados[k]["MAPE_%"])
    r = resultados[mejor]
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(r["test"]["ds"], r["test"]["y"], label="Real", color="#222", lw=1.8)
    for nombre, res in resultados.items():
        ax.plot(res["pred"]["ds"], res["pred"]["yhat"], lw=1.4, alpha=0.85,
                label=f"{nombre} (MAPE {res['MAPE_%']:.1f}%)")
    ax.set_title(f"Prophet + clima — mejor: {mejor} ({r['MAPE_%']:.2f}%)")
    ax.set_ylabel("Demanda (MW)")
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIGS / "prophet_clima_comparativa.png", dpi=110)
    print(f"\nFigura -> {FIGS / 'prophet_clima_comparativa.png'}")


if __name__ == "__main__":
    main()
