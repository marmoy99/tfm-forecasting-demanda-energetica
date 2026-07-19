"""
SARIMAX — Regresión armónica dinámica (dynamic harmonic regression).

POR QUÉ NO UN SARIMA CLÁSICO
La demanda eléctrica tiene TRES estacionalidades solapadas: diaria (24h),
semanal (168h) y anual (~8766h). SARIMAX solo tiene una ranura estacional (el
parámetro m), así que no puede con las tres. Y poner m=168 sobre 46.000
observaciones dispara el coste (el vector de estado crece con m): inviable.

La solución estándar (Hyndman) es la REGRESIÓN ARMÓNICA DINÁMICA:
  - Las estacionalidades salen FUERA del modelo, como exógenas en forma de
    senos y cosenos (términos de Fourier). Es el mismo truco que usa Prophet
    por dentro.
  - SARIMAX se queda solo con lo suyo: la autocorrelación de corto plazo (ARMA).
Resultado: SARIMAX(p,d,q) SIN parte estacional -> barato y con las 3 estacionalidades.

DOS TRAMPAS QUE COSTARON CARO (documentadas para la memoria):
  1. trend="c". SARIMAX en statsmodels NO pone constante por defecto. Sin ella,
     el modelo intenta explicar ~24.000 MW partiendo de cero. (MAPE 12% -> 9%)
  2. ESTANDARIZAR las exógenas. radiacion_solar vive en 0-950 y los senos de
     Fourier en [-1,1]: un factor 250 de diferencia. L-BFGS no converge con esas
     escalas y devuelve un ajuste malo SIN avisar. Estandarizando (con media/std
     del TRAIN, nunca del test), el MAPE cae de 12,4% a ~5% y el fit va 4x más
     rápido. Prophet estandariza sus regresores solo; SARIMAX no.
  (Se probó también modelar log(demanda) para emular la estacionalidad
   multiplicativa de Prophet: NO converge y empeora. Descartado.)

METODOLOGÍA (igual que prophet_tuning.py, para poder comparar de tú a tú):
    VALIDACIÓN  (bloque anterior)  -> se ELIGE el orden (p,d,q)
    TEST        (11 ventanas)      -> se MIDE; son las MISMAS ventanas que usa
                                      prophet_walkforward.py
"""
import logging
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

from prophet_baseline import (
    cargar_datos, metricas, HORIZON, HORIZON_DIAS, FIGS, ROOT,
)

warnings.filterwarnings("ignore")
logging.getLogger("statsmodels").setLevel(logging.ERROR)

CLIMA = ["HDD", "CDD", "humedad_relativa", "velocidad_viento", "radiacion_solar"]
# (periodo en horas, nº de armónicos). K alto = curva más detallada.
FOURIER = [(24, 8), (168, 3), (8766, 10)]
TREND = "c"          # constante SÍ; tendencia lineal NO (extrapolada se dispara)
ORDENES = [(1, 0, 0), (2, 0, 0), (3, 0, 1), (2, 0, 2)]

TEST_N, TEST_STEP = 12, 30   # mismas ventanas que el walk-forward de Prophet
VAL_N, VAL_STEP = 6, 60      # bloque anterior, para elegir el orden


# ----------------------------------------------------------------------------
def preparar_serie(df: pd.DataFrame) -> pd.DataFrame:
    """Rejilla horaria continua.

    SARIMAX asume observaciones equiespaciadas. El dataset va en hora local y le
    faltan las 02:00 de cada cambio a horario de verano (6 huecos en 5 años).
    Los rellenamos por interpolación: son 6 de 46.000 filas y ninguna cae dentro
    de una ventana de test.
    """
    s = df.set_index("datetime").sort_index()
    idx = pd.date_range(s.index.min(), s.index.max(), freq="h")
    n_huecos = len(idx) - len(s)
    s = s.reindex(idx).interpolate(method="time")
    print(f"Rejilla horaria continua: {n_huecos} horas rellenadas (huecos DST).")
    return s


def construir_exogenas(s: pd.DataFrame, con_clima: bool) -> pd.DataFrame:
    """Fourier (las 3 estacionalidades) + festivos [+ clima]."""
    idx = s.index
    # Horas transcurridas desde el origen: define la fase de cada sinusoide.
    t = (idx - idx[0]).total_seconds() / 3600.0
    cols = {}
    for periodo, K in FOURIER:
        for k in range(1, K + 1):
            cols[f"sin_{periodo}_{k}"] = np.sin(2 * np.pi * k * t / periodo)
            cols[f"cos_{periodo}_{k}"] = np.cos(2 * np.pi * k * t / periodo)
    X = pd.DataFrame(cols, index=idx)
    X["es_festivo"] = s["es_festivo"].to_numpy()
    if con_clima:
        for c in CLIMA:
            X[c] = s[c].to_numpy()
    return X


def evaluar_ventana(y, X, cutoff, order) -> dict | None:
    """Entrena hasta cutoff y predice las HORIZON horas siguientes."""
    ytr, yte = y.loc[:cutoff], y.loc[cutoff:].iloc[1:HORIZON + 1]
    if len(yte) < HORIZON:
        return None
    Xtr, Xte = X.loc[ytr.index], X.loc[yte.index]

    # Estandarización con estadísticos del TRAIN (evita leakage y hace converger)
    mu, sd = Xtr.mean(), Xtr.std().replace(0, 1)
    Xtr_s, Xte_s = (Xtr - mu) / sd, (Xte - mu) / sd

    t0 = time.time()
    modelo = SARIMAX(ytr, exog=Xtr_s, order=order, trend=TREND,
                     enforce_stationarity=False, enforce_invertibility=False)
    res = modelo.fit(disp=False, maxiter=300)
    pred = res.forecast(steps=HORIZON, exog=Xte_s)

    m = metricas(yte.to_numpy(), pred.to_numpy())
    m["converge"] = bool(res.mle_retvals.get("converged", False))
    m["segundos"] = round(time.time() - t0, 1)
    return m


# ----------------------------------------------------------------------------
def main():
    s = preparar_serie(cargar_datos())
    y = s["demanda_mw"]
    X_base = construir_exogenas(s, con_clima=False)
    X_clima = construir_exogenas(s, con_clima=True)
    print(f"Serie: {len(y)} horas | exógenas: {X_base.shape[1]} (sin clima) / "
          f"{X_clima.shape[1]} (con clima)\n")

    fin = y.index.max()
    test_cutoffs = [fin - pd.Timedelta(days=HORIZON_DIAS + TEST_STEP * i)
                    for i in range(TEST_N)][::-1]
    val_cutoffs = [test_cutoffs[0] - pd.Timedelta(days=VAL_STEP * (i + 1))
                   for i in range(VAL_N)][::-1]

    # ---- Etapa 1: elegir el orden (p,d,q) EN VALIDACIÓN ----
    print(f"=== Eligiendo orden ARMA en VALIDACIÓN "
          f"({val_cutoffs[0]:%Y-%m-%d} -> {val_cutoffs[-1]:%Y-%m-%d}) ===")
    filas_val = []
    for order in ORDENES:
        mapes = [r["MAPE_%"] for c in val_cutoffs
                 if (r := evaluar_ventana(y, X_clima, c, order)) is not None]
        mape = float(np.mean(mapes))
        filas_val.append({"order": str(order), "MAPE_val_%": round(mape, 2)})
        print(f"  SARIMAX{order}: MAPE val {mape:.2f}%")

    val = pd.DataFrame(filas_val).sort_values("MAPE_val_%")
    val.to_csv(ROOT / "reports" / "sarimax_validacion.csv", index=False)
    mejor = ORDENES[[str(o) for o in ORDENES].index(val.iloc[0]["order"])]
    print(f"\nOrden elegido: SARIMAX{mejor}\n")

    # ---- Etapa 2: medir en TEST (mismas ventanas que Prophet) ----
    print(f"=== TEST: {len(test_cutoffs)} ventanas "
          f"({test_cutoffs[0]:%Y-%m-%d} -> {test_cutoffs[-1]:%Y-%m-%d}) ===")
    filas = []
    for nombre, X in [("sarimax baseline", X_base), ("sarimax clima", X_clima)]:
        for cut in test_cutoffs:
            r = evaluar_ventana(y, X, cut, mejor)
            if r is None:
                continue
            r.update({"modelo": nombre, "cutoff": cut, "order": str(mejor)})
            filas.append(r)
            aviso = "" if r["converge"] else "  <-- NO CONVERGE"
            print(f"[{nombre:16}] {cut:%Y-%m-%d} -> MAPE {r['MAPE_%']:5.2f}% "
                  f"({r['segundos']:.0f}s){aviso}")

    res = pd.DataFrame(filas)
    res.to_csv(ROOT / "reports" / "sarimax_resultados.csv", index=False)

    print(f"\n=== SARIMAX (test, horizonte {HORIZON_DIAS} días) ===")
    resumen = res.groupby("modelo").agg(
        MAPE_medio=("MAPE_%", "mean"), MAPE_std=("MAPE_%", "std"),
        MAPE_max=("MAPE_%", "max"), MAE_medio=("MAE_MW", "mean"),
    ).round(2)
    print(resumen.to_string())
    if not res["converge"].all():
        print(f"\n AVISO: {(~res['converge']).sum()} ajustes no convergieron.")

    # ---- Comparación directa con Prophet (mismas ventanas) ----
    wf = ROOT / "reports" / "walkforward_resultados.csv"
    if wf.exists():
        p = pd.read_csv(wf, parse_dates=["cutoff"])
        p["modelo"] = "prophet " + p["modelo"]
        comp = pd.concat([p[["cutoff", "modelo", "MAPE_%"]],
                          res[["cutoff", "modelo", "MAPE_%"]]])
        print("\n=== PROPHET vs SARIMAX (mismas ventanas) ===")
        print(comp.groupby("modelo")["MAPE_%"].agg(["mean", "std"])
              .round(2).sort_values("mean").to_string())

        # Test pareado entre los dos mejores (clima vs clima)
        from scipy import stats
        piv = comp.pivot_table(index="cutoff", columns="modelo", values="MAPE_%")
        if {"prophet clima", "sarimax clima"} <= set(piv.columns):
            a, b = piv["prophet clima"].dropna(), piv["sarimax clima"].dropna()
            comun = a.index.intersection(b.index)
            a, b = a[comun], b[comun]
            gana = (a > b).sum()
            pv = stats.wilcoxon(a, b).pvalue
            print(f"\nSARIMAX gana a Prophet en {gana}/{len(comun)} ventanas "
                  f"| dif media {(a - b).mean():+.2f} pp | Wilcoxon p={pv:.4f}")
            print("  -> " + ("diferencia SIGNIFICATIVA (p<0.05)" if pv < 0.05
                             else "diferencia NO significativa: son equivalentes"))

        fig, ax = plt.subplots(figsize=(13, 5))
        for nombre in sorted(comp.modelo.unique()):
            sub = comp[comp.modelo == nombre]
            ax.plot(sub["cutoff"], sub["MAPE_%"], marker="o", lw=1.5, alpha=.85,
                    label=f"{nombre} ({sub['MAPE_%'].mean():.2f}%)")
        ax.axhline(3, color="green", ls="--", lw=1, label="Objetivo TFM (3%)")
        ax.set_title(f"Prophet vs SARIMAX armónico — MAPE a {HORIZON_DIAS} días")
        ax.set_ylabel("MAPE (%)")
        ax.set_xlabel("Fecha de corte (cutoff)")
        ax.legend(fontsize=8)
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(FIGS / "prophet_vs_sarimax.png", dpi=110)
        print(f"\nFigura -> {FIGS / 'prophet_vs_sarimax.png'}")


if __name__ == "__main__":
    main()
