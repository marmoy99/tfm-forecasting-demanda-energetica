"""
sarimax_walk_forward.py

Objetivo:
    Evaluar modelos SARIMAX sobre varios cortes temporales mediante una lógica
    walk-forward simplificada.

Salida:
    - Tabla comparativa de MAPE por corte y modelo.
    - CSV con los resultados agregados.

Modelos evaluados:
    1. SARIMAX base:
       variables meteorológicas + calendario + dummies de día de semana.

    2. SARIMAX + lag 168h:
       variables meteorológicas + calendario + dummies de día de semana
       + demanda observada en la misma hora de la semana anterior.

    3. Baseline lag 24h:
       predicción ingenua usando la demanda de la misma hora del día anterior.

    4. Baseline lag 168h:
       predicción ingenua usando la demanda de la misma hora de la semana anterior.

Nota metodológica:
    La variante con demanda_lag_168h se considera experimental hasta comprobar
    si la mejora observada en un único corte se mantiene en varios cortes
    temporales. Esta variable es razonablemente defendible para horizontes de
    hasta 7 días, ya que usa información observada una semana antes del periodo
    predicho.
"""

import os
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


# =============================================================================
# 1. Definición de rutas
# =============================================================================

CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))

# Subimos desde src/models/SARIMAX hasta la raíz del repositorio.
RAIZ_REPOSITORIO = os.path.abspath(os.path.join(CARPETA_SCRIPT, "..", "..", ".."))

FICHERO = os.path.join(
    RAIZ_REPOSITORIO,
    "Trabajo Miguel",
    "data",
    "processed",
    "dataset_modelado.csv",
)

FICHERO_RESULTADOS = os.path.join(
    CARPETA_SCRIPT,
    "resultados_sarimax_walk_forward.csv",
)


# =============================================================================
# 2. Parámetros editables del experimento
# =============================================================================

REGRESORES_BASE = [
    "HDD",
    "CDD",
    "humedad_relativa",
    "velocidad_viento",
    "radiacion_solar",
    "es_festivo",
    "es_fin_de_semana",
    "es_agosto",
]

COLUMNAS_NECESARIAS = [
    "demanda_mw",
    "dia_semana",
    "demanda_lag_24h",
    "demanda_lag_168h",
] + REGRESORES_BASE

# Cortes equivalentes a los usados en la evaluación walk-forward de Prophet.
CORTES = [
    "2025-02-05",
    "2025-05-06",
    "2025-07-08",
    "2025-11-04",
]

DIAS_TEST = 3
TRAIN_DIAS = 365

# Configuración inicial conservadora, alineada con sarimax_evaluacion.py.
ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 0, 0, 24)
MAXITER = 25

VARIANTES_SARIMAX = [
    {
        "modelo": "SARIMAX base",
        "usar_lag_168h": False,
    },
    {
        "modelo": "SARIMAX + lag 168h",
        "usar_lag_168h": True,
    },
]


# =============================================================================
# 3. Funciones auxiliares
# =============================================================================

def calcular_mape(real: pd.Series, estimado: pd.Series) -> float:
    """
    Calcula el MAPE en porcentaje.

    Se asume que la demanda real no contiene ceros, lo cual es razonable
    para demanda eléctrica peninsular horaria.
    """
    real, estimado = real.align(estimado, join="inner")

    if real.empty:
        return np.nan

    return (np.abs(real - estimado) / real).mean() * 100


def validar_columnas(df: pd.DataFrame, columnas_necesarias: list[str]) -> None:
    """
    Comprueba que el dataset contiene todas las columnas necesarias.
    """
    columnas_faltantes = [col for col in columnas_necesarias if col not in df.columns]

    if columnas_faltantes:
        raise ValueError(f"Faltan columnas en el dataset: {columnas_faltantes}")


def cargar_dataset(ruta: str) -> pd.DataFrame:
    """
    Carga el dataset de modelado y prepara el índice temporal.

    No se fuerza frecuencia horaria con asfreq("h") porque en series horarias
    locales pueden existir saltos asociados al cambio horario. Forzar la
    frecuencia puede crear filas artificiales con NaN, por ejemplo durante
    el cambio al horario de verano.
    """
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el dataset en la ruta: {ruta}")

    df = pd.read_csv(ruta)

    if "datetime" not in df.columns:
        raise ValueError("El dataset debe contener una columna llamada 'datetime'.")

    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    df = df.sort_index()

    validar_columnas(df, COLUMNAS_NECESARIAS)

    return df


def preparar_train_test(
    df: pd.DataFrame,
    fecha_corte: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide el dataset en entrenamiento y test para un corte temporal concreto.
    """
    inicio_test = pd.Timestamp(fecha_corte)
    fin_test = inicio_test + pd.Timedelta(days=DIAS_TEST)
    inicio_train = inicio_test - pd.Timedelta(days=TRAIN_DIAS)

    train = df[(df.index >= inicio_train) & (df.index < inicio_test)].copy()
    test = df[(df.index >= inicio_test) & (df.index < fin_test)].copy()

    if train.empty:
        raise ValueError(
            f"El conjunto de entrenamiento quedó vacío para el corte {fecha_corte}."
        )

    if test.empty:
        raise ValueError(
            f"El conjunto de test quedó vacío para el corte {fecha_corte}."
        )

    return train, test


def validar_sin_nulos_para_modelo(
    y_train: pd.Series,
    exog_train: pd.DataFrame,
    y_test: pd.Series,
    exog_test: pd.DataFrame,
) -> None:
    """
    Detiene la ejecución si existen nulos o infinitos en las matrices del modelo.
    """
    objetos = {
        "y_train": y_train,
        "exog_train": exog_train,
        "y_test": y_test,
        "exog_test": exog_test,
    }

    for nombre, datos in objetos.items():
        if datos.isna().sum().sum() > 0:
            raise ValueError(f"{nombre} contiene valores nulos.")

        if isinstance(datos, pd.DataFrame):
            datos_numericos = datos.select_dtypes(include=[np.number])
        else:
            datos_numericos = datos

        if np.isinf(datos_numericos).sum().sum() > 0:
            raise ValueError(f"{nombre} contiene valores infinitos.")


def preparar_exog(datos: pd.DataFrame, usar_lag_168h: bool) -> pd.DataFrame:
    """
    Prepara variables exógenas para SARIMAX.

    Incluye:
        - Variables meteorológicas.
        - Variables básicas de calendario.
        - Dummies de día de semana.
        - Opcionalmente, demanda_lag_168h.
    """
    exog = datos[REGRESORES_BASE].copy()

    if usar_lag_168h:
        exog["demanda_lag_168h"] = datos["demanda_lag_168h"]

    dummies_dia = pd.get_dummies(
        datos["dia_semana"],
        prefix="dia_semana",
        drop_first=True,
        dtype=float,
    )

    exog = pd.concat([exog, dummies_dia], axis=1)

    return exog.astype(float)


def preparar_matrices_modelo(
    train: pd.DataFrame,
    test: pd.DataFrame,
    usar_lag_168h: bool,
) -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Construye las matrices endógenas y exógenas para SARIMAX.

    La preparación conjunta de variables exógenas garantiza que train y test
    tengan exactamente las mismas columnas cuando se generan dummies.
    """
    datos_completos = pd.concat([train, test], axis=0)

    exog_completa = preparar_exog(
        datos=datos_completos,
        usar_lag_168h=usar_lag_168h,
    )

    exog_train = exog_completa.loc[train.index]
    exog_test = exog_completa.loc[test.index]

    y_train = train["demanda_mw"].astype(float)
    y_test = test["demanda_mw"].astype(float)

    validar_sin_nulos_para_modelo(y_train, exog_train, y_test, exog_test)

    return y_train, exog_train, y_test, exog_test


def entrenar_sarimax(y_train: pd.Series, exog_train: pd.DataFrame):
    """
    Entrena un modelo SARIMAX con índice numérico interno.

    Las fechas se conservan fuera del modelo para evaluación, pero statsmodels
    recibe índices consecutivos para evitar problemas con índices temporales
    locales sin frecuencia explícita.
    """
    y_modelo = y_train.copy()
    exog_modelo = exog_train.copy()

    y_modelo.index = pd.RangeIndex(start=0, stop=len(y_modelo))
    exog_modelo.index = pd.RangeIndex(start=0, stop=len(exog_modelo))

    modelo = SARIMAX(
        endog=y_modelo,
        exog=exog_modelo,
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resultado = modelo.fit(disp=False, maxiter=MAXITER)

    return resultado


def predecir(resultado, exog_test: pd.DataFrame) -> pd.Series:
    """
    Genera predicciones SARIMAX para el periodo de test.
    """
    indice_temporal = exog_test.index

    exog_modelo = exog_test.copy()
    exog_modelo.index = pd.RangeIndex(start=0, stop=len(exog_modelo))

    pred = resultado.get_forecast(
        steps=len(exog_modelo),
        exog=exog_modelo,
    )

    estimado = pred.predicted_mean
    estimado.index = indice_temporal

    return estimado


def evaluar_baseline_lag(test: pd.DataFrame, columna_lag: str) -> float:
    """
    Evalúa un baseline ingenuo basado en una columna lag del dataset.
    """
    if columna_lag not in test.columns:
        return np.nan

    datos = test[["demanda_mw", columna_lag]].dropna()

    if datos.empty:
        return np.nan

    return calcular_mape(
        real=datos["demanda_mw"],
        estimado=datos[columna_lag],
    )


def evaluar_sarimax_en_corte(
    train: pd.DataFrame,
    test: pd.DataFrame,
    fecha_corte: str,
    nombre_modelo: str,
    usar_lag_168h: bool,
) -> dict:
    """
    Entrena y evalúa una variante SARIMAX en un corte concreto.
    """
    y_train, exog_train, y_test, exog_test = preparar_matrices_modelo(
        train=train,
        test=test,
        usar_lag_168h=usar_lag_168h,
    )

    resultado = entrenar_sarimax(
        y_train=y_train,
        exog_train=exog_train,
    )

    estimado = predecir(
        resultado=resultado,
        exog_test=exog_test,
    )

    mape = calcular_mape(y_test, estimado)

    return {
        "corte": fecha_corte,
        "modelo": nombre_modelo,
        "tipo_modelo": "SARIMAX",
        "usar_lag_168h": usar_lag_168h,
        "mape": mape,
        "filas_train": len(train),
        "filas_test": len(test),
        "train_dias": TRAIN_DIAS,
        "dias_test": DIAS_TEST,
        "order": str(ORDER),
        "seasonal_order": str(SEASONAL_ORDER),
        "error": "",
    }


def evaluar_baselines_en_corte(test: pd.DataFrame, fecha_corte: str) -> list[dict]:
    """
    Evalúa los baselines ingenuos de 24h y 168h en un corte concreto.
    """
    resultados = []

    baselines = [
        {
            "modelo": "Baseline lag 24h",
            "columna": "demanda_lag_24h",
        },
        {
            "modelo": "Baseline lag 168h",
            "columna": "demanda_lag_168h",
        },
    ]

    for baseline in baselines:
        mape = evaluar_baseline_lag(
            test=test,
            columna_lag=baseline["columna"],
        )

        resultados.append({
            "corte": fecha_corte,
            "modelo": baseline["modelo"],
            "tipo_modelo": "baseline_naive",
            "usar_lag_168h": "",
            "mape": mape,
            "filas_train": "",
            "filas_test": len(test),
            "train_dias": TRAIN_DIAS,
            "dias_test": DIAS_TEST,
            "order": "",
            "seasonal_order": "",
            "error": "",
        })

    return resultados


def imprimir_tabla_resultados(resultados: pd.DataFrame) -> None:
    """
    Imprime una tabla resumen de MAPE por corte y modelo.
    """
    tabla = resultados.pivot_table(
        index="corte",
        columns="modelo",
        values="mape",
        aggfunc="first",
    )

    print("\nResultados walk-forward SARIMAX")
    print("=" * 80)
    print(tabla.round(2).to_string())

    print("\nMAPE medio por modelo")
    print("=" * 80)
    print(
        resultados
        .groupby("modelo")["mape"]
        .mean()
        .sort_values()
        .round(2)
        .to_string()
    )


# =============================================================================
# 4. Ejecución principal
# =============================================================================

def main() -> None:
    """
    Ejecuta la evaluación walk-forward de SARIMAX.
    """
    print("Cargando dataset...")
    df = cargar_dataset(FICHERO)

    print("Iniciando walk-forward SARIMAX...")
    print(f"Cortes: {CORTES}")
    print(f"Ventana de entrenamiento: {TRAIN_DIAS} días")
    print(f"Horizonte test: {DIAS_TEST} días")
    print(f"ORDER: {ORDER}")
    print(f"SEASONAL_ORDER: {SEASONAL_ORDER}")

    resultados = []

    for fecha_corte in CORTES:
        print("\n" + "-" * 80)
        print(f"Corte temporal: {fecha_corte}")

        try:
            train, test = preparar_train_test(
                df=df,
                fecha_corte=fecha_corte,
            )

            print(f"Filas train: {len(train)}")
            print(f"Filas test: {len(test)}")

            resultados.extend(
                evaluar_baselines_en_corte(
                    test=test,
                    fecha_corte=fecha_corte,
                )
            )

            for variante in VARIANTES_SARIMAX:
                print(f"Entrenando {variante['modelo']}...")

                resultado_variante = evaluar_sarimax_en_corte(
                    train=train,
                    test=test,
                    fecha_corte=fecha_corte,
                    nombre_modelo=variante["modelo"],
                    usar_lag_168h=variante["usar_lag_168h"],
                )

                resultados.append(resultado_variante)

                print(
                    f"{variante['modelo']} | "
                    f"MAPE: {resultado_variante['mape']:.2f}%"
                )

        except Exception as exc:
            print(f"Error en corte {fecha_corte}: {exc}")

            resultados.append({
                "corte": fecha_corte,
                "modelo": "ERROR",
                "tipo_modelo": "error",
                "usar_lag_168h": "",
                "mape": np.nan,
                "filas_train": "",
                "filas_test": "",
                "train_dias": TRAIN_DIAS,
                "dias_test": DIAS_TEST,
                "order": str(ORDER),
                "seasonal_order": str(SEASONAL_ORDER),
                "error": str(exc),
            })

    resultados_df = pd.DataFrame(resultados)

    imprimir_tabla_resultados(resultados_df)

    resultados_df.to_csv(FICHERO_RESULTADOS, index=False, encoding="utf-8-sig")

    print(f"\nResultados guardados en: {FICHERO_RESULTADOS}")


if __name__ == "__main__":
    main()
