"""
sarimax_comparacion_clasico_armonico.py

Compara tres variantes SARIMAX bajo un protocolo común:
1. SARIMAX clásico + lag 168h
2. SARIMAX armónico
3. SARIMAX armónico + lag 168h

Salida:
    reports/model_comparison/resultados_sarimax_comparacion.csv

Columnas:
    corte, modelo, mape, mae, rmse
"""

from __future__ import annotations

import os
import time
import warnings
from dataclasses import dataclass

import holidays
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


# =============================================================================
# 1. Rutas
# =============================================================================

CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))
RAIZ_REPOSITORIO = os.path.abspath(
    os.path.join(CARPETA_SCRIPT, "..", "..", "..")
)

FICHERO_DATOS = os.path.join(
    RAIZ_REPOSITORIO,
    "data",
    "processed",
    "dataset_modelado.csv",
)

CARPETA_RESULTADOS = os.path.join(
    RAIZ_REPOSITORIO,
    "reports",
    "model_comparison",
)

FICHERO_RESULTADOS = os.path.join(
    CARPETA_RESULTADOS,
    "resultados_sarimax_comparacion.csv",
)


# =============================================================================
# 2. Configuración común
# =============================================================================

REGRESORES_METEO = [
    "HDD",
    "CDD",
    "humedad_relativa",
    "velocidad_viento",
    "radiacion_solar",
]

DIAS_TRAIN = 365
HORAS_TEST = 72
N_CORTES = 12
SEPARACION_CORTES_DIAS = 30

ORDER_CLASICO = (1, 1, 1)
TREND_CLASICO = None
MAXITER = 300

ORDER_ARMONICO = (2, 0, 2)
TREND_ARMONICO = "c"
MAXITER = 300

SEASONAL_ORDER_CLASICO = (1, 0, 0, 24)
SEASONAL_ORDER_ARMONICO = (0, 0, 0, 0)

FOURIER = [
    (24, 6),
    (168, 3),
    (8766, 3),
]


@dataclass(frozen=True)
class ConfiguracionModelo:
    nombre: str
    usar_fourier: bool
    usar_lag_168h: bool
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    trend: str | None


CONFIGURACIONES = [
    ConfiguracionModelo(
        nombre="SARIMAX clásico + lag 168h",
        usar_fourier=False,
        usar_lag_168h=True,
        order=ORDER_CLASICO,
        seasonal_order=SEASONAL_ORDER_CLASICO,
        trend=TREND_CLASICO,
    ),
    ConfiguracionModelo(
        nombre="SARIMAX armónico",
        usar_fourier=True,
        usar_lag_168h=False,
        order=ORDER_ARMONICO,
        seasonal_order=SEASONAL_ORDER_ARMONICO,
        trend=TREND_ARMONICO,
    ),
    ConfiguracionModelo(
        nombre="SARIMAX armónico + lag 168h",
        usar_fourier=True,
        usar_lag_168h=True,
        order=ORDER_ARMONICO,
        seasonal_order=SEASONAL_ORDER_ARMONICO,
        trend=TREND_ARMONICO,
    ),
]


# =============================================================================
# 3. Preparación común del dataset
# =============================================================================

def validar_columnas(df: pd.DataFrame, columnas: list[str]) -> None:
    faltantes = [col for col in columnas if col not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")


def cargar_y_preparar_dataset(ruta: str) -> pd.DataFrame:
    """
    Carga el dataset, crea una rejilla horaria continua, interpola demanda y
    meteorología, regenera calendario y calcula demanda_lag_168h.
    """
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el dataset: {ruta}")

    df = pd.read_csv(ruta, parse_dates=["datetime"])
    validar_columnas(df, ["datetime", "demanda_mw"] + REGRESORES_METEO)

    df = df.set_index("datetime").sort_index()

    if df.index.has_duplicates:
        raise ValueError(
            f"El índice contiene {df.index.duplicated().sum()} timestamps duplicados."
        )

    indice_completo = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq="h",
    )

    huecos = len(indice_completo) - len(df)
    df = df.reindex(indice_completo)

    columnas_interpolar = ["demanda_mw"] + REGRESORES_METEO
    df[columnas_interpolar] = (
        df[columnas_interpolar]
        .interpolate(method="time", limit_direction="both")
    )

    calendario_es = holidays.Spain(
        years=range(df.index.min().year, df.index.max().year + 1)
    )

    df["mes"] = df.index.month
    df["hora"] = df.index.hour
    df["dia_semana"] = df.index.dayofweek
    df["es_fin_de_semana"] = df["dia_semana"].isin([5, 6]).astype(int)
    df["es_agosto"] = (df["mes"] == 8).astype(int)
    df["es_festivo"] = df.index.map(
        lambda fecha: int(fecha.date() in calendario_es)
    )

    df["demanda_lag_168h"] = df["demanda_mw"].shift(168)
    df = df.dropna(subset=["demanda_lag_168h"]).copy()

    columnas_validar = (
        ["demanda_mw", "demanda_lag_168h"]
        + REGRESORES_METEO
        + ["es_festivo", "es_fin_de_semana", "es_agosto", "dia_semana"]
    )

    nulos = (
        df[columnas_validar]
        .replace([np.inf, -np.inf], np.nan)
        .isna()
        .sum()
    )
    nulos = nulos[nulos > 0]

    if not nulos.empty:
        raise ValueError(f"Persisten nulos o infinitos:\n{nulos}")

    print(
        f"Dataset preparado: {len(df):,} horas | "
        f"huecos interpolados: {huecos}"
    )

    return df


# =============================================================================
# 4. Cortes y ventanas
# =============================================================================

def construir_cortes(df: pd.DataFrame) -> list[pd.Timestamp]:
    """
    Mismo criterio usado en los walk-forward de Prophet y LightGBM:
    doce cortes separados por 30 días, construidos desde el final.
    """
    fin = df.index.max()

    cortes = [
        fin - pd.Timedelta(
            days=(HORAS_TEST / 24) + SEPARACION_CORTES_DIAS * i
        )
        for i in range(N_CORTES)
    ]

    return cortes[::-1]


def preparar_train_test(
    df: pd.DataFrame,
    corte: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    inicio_train = corte - pd.Timedelta(days=DIAS_TRAIN)
    fin_test = corte + pd.Timedelta(hours=HORAS_TEST)

    train = df[(df.index >= inicio_train) & (df.index < corte)].copy()
    test = df[(df.index >= corte) & (df.index < fin_test)].copy()

    if len(train) != DIAS_TRAIN * 24:
        raise ValueError(
            f"Train incompleto en {corte}: {len(train)} filas."
        )

    if len(test) != HORAS_TEST:
        raise ValueError(
            f"Test incompleto en {corte}: {len(test)} filas."
        )

    return train, test


# =============================================================================
# 5. Variables exógenas
# =============================================================================

def construir_fourier(
    indice: pd.DatetimeIndex,
    origen: pd.Timestamp,
) -> pd.DataFrame:
    horas = (indice - origen).total_seconds() / 3600.0
    columnas = {}

    for periodo, k_max in FOURIER:
        for k in range(1, k_max + 1):
            columnas[f"sin_{periodo}_{k}"] = np.sin(
                2 * np.pi * k * horas / periodo
            )
            columnas[f"cos_{periodo}_{k}"] = np.cos(
                2 * np.pi * k * horas / periodo
            )

    return pd.DataFrame(columnas, index=indice)


def construir_exogenas(
    datos: pd.DataFrame,
    configuracion: ConfiguracionModelo,
    origen_fourier: pd.Timestamp,
) -> pd.DataFrame:
    exog = datos[
        REGRESORES_METEO
        + ["es_festivo", "es_fin_de_semana", "es_agosto"]
    ].copy()

    dummies_dia = pd.get_dummies(
        datos["dia_semana"],
        prefix="dia_semana",
        drop_first=True,
        dtype=float,
    )

    exog = pd.concat([exog, dummies_dia], axis=1)

    if configuracion.usar_fourier:
        exog = pd.concat(
            [
                exog,
                construir_fourier(
                    indice=datos.index,
                    origen=origen_fourier,
                ),
            ],
            axis=1,
        )

    if configuracion.usar_lag_168h:
        exog["demanda_lag_168h"] = datos["demanda_lag_168h"]

    return exog.astype(float)


def estandarizar_exogenas(
    exog_train: pd.DataFrame,
    exog_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    media = exog_train.mean()
    desviacion = exog_train.std().replace(0, 1)

    return (
        (exog_train - media) / desviacion,
        (exog_test - media) / desviacion,
    )


def preparar_matrices(
    train: pd.DataFrame,
    test: pd.DataFrame,
    configuracion: ConfiguracionModelo,
) -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    datos_completos = pd.concat([train, test], axis=0)

    exog_completa = construir_exogenas(
        datos=datos_completos,
        configuracion=configuracion,
        origen_fourier=train.index.min(),
    )

    exog_train = exog_completa.loc[train.index]
    exog_test = exog_completa.loc[test.index]

    exog_train, exog_test = estandarizar_exogenas(
        exog_train,
        exog_test,
    )

    y_train = train["demanda_mw"].astype(float)
    y_test = test["demanda_mw"].astype(float)

    for nombre, objeto in {
        "y_train": y_train,
        "y_test": y_test,
        "exog_train": exog_train,
        "exog_test": exog_test,
    }.items():
        if objeto.replace([np.inf, -np.inf], np.nan).isna().sum().sum() > 0:
            raise ValueError(f"{nombre} contiene nulos o infinitos.")

    return y_train, exog_train, y_test, exog_test


# =============================================================================
# 6. Modelo y métricas
# =============================================================================

def calcular_metricas(
    real: pd.Series,
    estimado: pd.Series,
) -> tuple[float, float, float]:
    real, estimado = real.align(estimado, join="inner")

    if real.empty:
        raise ValueError("No hay datos alineados para calcular métricas.")

    if (real == 0).any():
        raise ValueError("MAPE no admite valores reales iguales a cero.")

    error = real - estimado

    mape = float((np.abs(error) / real).mean() * 100)
    mae = float(np.abs(error).mean())
    rmse = float(np.sqrt(np.mean(error**2)))

    return mape, mae, rmse


def entrenar_y_predecir(
    y_train: pd.Series,
    exog_train: pd.DataFrame,
    exog_test: pd.DataFrame,
    configuracion: ConfiguracionModelo,
) -> tuple[pd.Series, bool, float]:
    y_modelo = y_train.reset_index(drop=True)
    x_train_modelo = exog_train.reset_index(drop=True)
    x_test_modelo = exog_test.reset_index(drop=True)

    modelo = SARIMAX(
        endog=y_modelo,
        exog=x_train_modelo,
        order=configuracion.order,
        seasonal_order=configuracion.seasonal_order,
        trend=configuracion.trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    inicio = time.time()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resultado = modelo.fit(
            disp=False,
            maxiter=MAXITER,
        )

    segundos = time.time() - inicio
    converge = bool(resultado.mle_retvals.get("converged", False))

    prediccion = resultado.get_forecast(
        steps=len(x_test_modelo),
        exog=x_test_modelo,
    ).predicted_mean

    prediccion.index = exog_test.index

    return prediccion, converge, segundos


def evaluar_modelo_en_corte(
    train: pd.DataFrame,
    test: pd.DataFrame,
    corte: pd.Timestamp,
    configuracion: ConfiguracionModelo,
) -> dict:
    y_train, exog_train, y_test, exog_test = preparar_matrices(
        train,
        test,
        configuracion,
    )

    prediccion, converge, segundos = entrenar_y_predecir(
        y_train,
        exog_train,
        exog_test,
        configuracion,
    )

    mape, mae, rmse = calcular_metricas(y_test, prediccion)

    aviso = "" if converge else " | AVISO: NO CONVERGE"

    print(
        f"{configuracion.nombre:34} | "
        f"MAPE {mape:6.2f}% | "
        f"MAE {mae:8.2f} MW | "
        f"RMSE {rmse:8.2f} MW | "
        f"{segundos:6.1f}s"
        f"{aviso}"
    )

    return {
        "corte": corte,
        "modelo": configuracion.nombre,
        "mape": mape,
        "mae": mae,
        "rmse": rmse,
    }


def imprimir_resumen(resultados: pd.DataFrame) -> None:
    resumen = (
        resultados
        .groupby("modelo")
        .agg(
            mape_medio=("mape", "mean"),
            mape_std=("mape", "std"),
            mae_medio=("mae", "mean"),
            rmse_medio=("rmse", "mean"),
        )
        .sort_values("mape_medio")
        .round(2)
    )

    print("\n" + "=" * 95)
    print("RESUMEN COMPARATIVO")
    print("=" * 95)
    print(resumen.to_string())


# =============================================================================
# 7. Ejecución
# =============================================================================

def main() -> None:
    """
    Ejecuta la comparación walk-forward entre las tres variantes SARIMAX
    bajo un protocolo temporal común.
    """
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)

    print("Cargando y preparando dataset...")
    df = cargar_y_preparar_dataset(FICHERO_DATOS)

    print("Construyendo cortes temporales...")
    cortes = construir_cortes(df)

    print("\nProtocolo común")
    print("-" * 95)
    print(f"Cortes: {len(cortes)}")
    print(f"Ventana de entrenamiento: {DIAS_TRAIN} días")
    print(f"Horizonte de test: {HORAS_TEST} horas")
    print(f"Separación entre cortes: {SEPARACION_CORTES_DIAS} días")
    print(f"Máximo de iteraciones: {MAXITER}")

    print("\nConfiguración SARIMAX clásico")
    print("-" * 95)
    print(f"ORDER: {ORDER_CLASICO}")
    print(f"SEASONAL_ORDER: {SEASONAL_ORDER_CLASICO}")
    print(f"TREND: {TREND_CLASICO}")
    print("Lag 168h: Sí")

    print("\nConfiguración SARIMAX armónico")
    print("-" * 95)
    print(f"ORDER: {ORDER_ARMONICO}")
    print(f"SEASONAL_ORDER: {SEASONAL_ORDER_ARMONICO}")
    print(f"TREND: {TREND_ARMONICO}")
    print(f"Fourier: {FOURIER}")

    print("\nRango temporal de evaluación")
    print("-" * 95)
    print(f"Primer corte: {cortes[0]}")
    print(f"Último corte: {cortes[-1]}")

    resultados = []

    for numero, corte in enumerate(cortes, start=1):
        print("\n" + "=" * 95)
        print(f"CORTE {numero}/{len(cortes)}: {corte}")
        print("=" * 95)

        train, test = preparar_train_test(
            df=df,
            corte=corte,
        )

        print(
            f"Train: {train.index.min()} -> {train.index.max()} "
            f"({len(train)} horas)"
        )
        print(
            f"Test:  {test.index.min()} -> {test.index.max()} "
            f"({len(test)} horas)"
        )

        for configuracion in CONFIGURACIONES:
            print("\n" + "-" * 95)
            print(f"Modelo: {configuracion.nombre}")
            print(f"ORDER: {configuracion.order}")
            print(f"SEASONAL_ORDER: {configuracion.seasonal_order}")
            print(f"TREND: {configuracion.trend}")
            print(f"Usa Fourier: {configuracion.usar_fourier}")
            print(f"Usa lag 168h: {configuracion.usar_lag_168h}")

            try:
                resultado = evaluar_modelo_en_corte(
                    train=train,
                    test=test,
                    corte=corte,
                    configuracion=configuracion,
                )

                resultados.append(resultado)

            except Exception as exc:
                raise RuntimeError(
                    f"Error en el corte {corte} con el modelo "
                    f"'{configuracion.nombre}': {exc}"
                ) from exc

    resultados_df = pd.DataFrame(
        resultados,
        columns=[
            "corte",
            "modelo",
            "mape",
            "mae",
            "rmse",
        ],
    )

    if resultados_df.empty:
        raise RuntimeError(
            "La ejecución terminó sin generar resultados."
        )

    filas_esperadas = len(cortes) * len(CONFIGURACIONES)

    if len(resultados_df) != filas_esperadas:
        raise RuntimeError(
            "El número de resultados generados no coincide con el esperado. "
            f"Generados: {len(resultados_df)}. "
            f"Esperados: {filas_esperadas}."
        )

    resultados_df = (
        resultados_df
        .sort_values(["corte", "modelo"])
        .reset_index(drop=True)
    )

    resultados_df.to_csv(
        FICHERO_RESULTADOS,
        index=False,
        encoding="utf-8-sig",
    )

    imprimir_resumen(resultados_df)

    print("\n" + "=" * 95)
    print("COMPARACIÓN FINALIZADA")
    print("=" * 95)
    print(f"CSV guardado en: {FICHERO_RESULTADOS}")
    print(f"Filas generadas: {len(resultados_df)}")
    print(f"Filas esperadas: {filas_esperadas}")
    print(f"Columnas: {list(resultados_df.columns)}")


if __name__ == "__main__":
    main()
