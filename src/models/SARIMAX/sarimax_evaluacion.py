"""
sarimax_evaluacion.py

Objetivo:
    Entrenar y evaluar un primer modelo SARIMAX sobre el dataset de demanda
    eléctrica horaria, usando un corte temporal simple comparable con
    el script prophet_evaluacion.py.

Salida:
    - MAPE del periodo de test.
    - Gráfico comparativo demanda real vs predicción SARIMAX.

Nota metodológica:
    En esta primera versión no se usan demanda_lag_24h ni demanda_lag_168h
    como variables exógenas para evitar riesgo de leakage en predicción futura.

    Se incorporan variables dummy de día de semana para reforzar la estructura
    semanal de la demanda, especialmente relevante para diferenciar días
    laborables, sábados y domingos.
"""

import os
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX


# =============================================================================
# 1. Definición de rutas
# =============================================================================

CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))

# Subimos desde src/models/SARIMAX hasta la raíz del repositorio.
# Esta lógica mantiene rutas relativas al script, igual que en los archivos Prophet,
# pero evita depender del directorio desde el que se ejecute el comando.
RAIZ_REPOSITORIO = os.path.abspath(os.path.join(CARPETA_SCRIPT, "..", "..", ".."))

FICHERO = os.path.join(
    RAIZ_REPOSITORIO,
    "Trabajo Miguel",
    "data",
    "processed",
    "dataset_modelado.csv",
)

CARPETA_GRAFICAS = os.path.join(CARPETA_SCRIPT, "imagenes_generadas")
os.makedirs(CARPETA_GRAFICAS, exist_ok=True)

FICHERO_GRAFICA = os.path.join(CARPETA_GRAFICAS, "sarimax_evaluacion.png")


# =============================================================================
# 2. Parámetros editables del experimento
# =============================================================================

# Variables exógenas iniciales.
# No incluimos lags de demanda para evitar leakage en la evaluación inicial.
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

# Variante experimental.
# Si True, SARIMAX incorpora demanda_lag_168h como regresor exógeno.
# Esta variable representa la demanda observada en la misma hora de la semana anterior.
USAR_LAG_168H = True

# Columnas necesarias para ejecutar el experimento.
COLUMNAS_NECESARIAS = [
    "demanda_mw",
    "dia_semana",
    "demanda_lag_24h",
    "demanda_lag_168h",
] + REGRESORES_BASE

FECHA_CORTE = "2025-10-03"
DIAS_TEST = 3

# Ventana de entrenamiento acotada para controlar coste computacional.
# Con 365 días el modelo ve un ciclo anual completo, aunque el entrenamiento
# sigue siendo razonablemente manejable frente a usar todo el histórico.
TRAIN_DIAS = 365

INTERVALO_HORAS_GRAFICA = 6

# Configuración inicial conservadora.
# La estacionalidad diaria se modela con periodo 24.
# La estructura semanal se refuerza mediante dummies de dia_semana.
ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 0, 0, 24)
MAXITER = 25


# =============================================================================
# 3. Funciones auxiliares
# =============================================================================

def calcular_mape(real: pd.Series, estimado: pd.Series) -> float:
    """
    Calcula el MAPE en porcentaje.

    Nota:
        Se asume que la demanda real no contiene ceros, lo cual es razonable
        para demanda eléctrica peninsular horaria.
    """
    real, estimado = real.align(estimado, join="inner")
    return (np.abs(real - estimado) / real).mean() * 100


def evaluar_baselines_naive(test: pd.DataFrame) -> tuple[pd.Series | None, pd.Series | None]:
    """
    Evalúa baselines ingenuos basados en lags ya presentes en el dataset.

    Devuelve:
        - pred_lag_24h
        - pred_lag_168h

    Estos baselines permiten contextualizar el rendimiento de SARIMAX:
        - demanda_lag_24h: misma hora del día anterior.
        - demanda_lag_168h: misma hora de la semana anterior.
    """
    print("\nEvaluación de baselines ingenuos")
    print("-" * 60)

    pred_lag_24h = None
    pred_lag_168h = None

    if "demanda_lag_24h" in test.columns:
        datos_lag_24 = test[["demanda_mw", "demanda_lag_24h"]].dropna()

        if not datos_lag_24.empty:
            pred_lag_24h = datos_lag_24["demanda_lag_24h"]
            mape_lag_24 = calcular_mape(
                datos_lag_24["demanda_mw"],
                pred_lag_24h,
            )
            print(f"MAPE baseline lag 24h: {mape_lag_24:.2f}%")
        else:
            print("No hay datos válidos para evaluar demanda_lag_24h.")
    else:
        print("La columna demanda_lag_24h no existe en el dataset.")

    if "demanda_lag_168h" in test.columns:
        datos_lag_168 = test[["demanda_mw", "demanda_lag_168h"]].dropna()

        if not datos_lag_168.empty:
            pred_lag_168h = datos_lag_168["demanda_lag_168h"]
            mape_lag_168 = calcular_mape(
                datos_lag_168["demanda_mw"],
                pred_lag_168h,
            )
            print(f"MAPE baseline lag 168h: {mape_lag_168:.2f}%")
        else:
            print("No hay datos válidos para evaluar demanda_lag_168h.")
    else:
        print("La columna demanda_lag_168h no existe en el dataset.")

    return pred_lag_24h, pred_lag_168h


def evaluar_mape_por_dia(
    real: pd.Series,
    estimado: pd.Series,
    nombre_modelo: str,
) -> None:
    """
    Calcula el MAPE segmentado por día natural.

    Esta evaluación permite detectar si el error del modelo se concentra
    en determinados días, por ejemplo fines de semana o festivos.
    """
    comparacion = pd.DataFrame({
        "real": real,
        "estimado": estimado,
    }).dropna()

    comparacion["fecha"] = comparacion.index.date

    print(f"\nMAPE segmentado por día - {nombre_modelo}")
    print("-" * 60)

    for fecha, grupo in comparacion.groupby("fecha"):
        mape_dia = calcular_mape(grupo["real"], grupo["estimado"])
        print(f"{fecha}: {mape_dia:.2f}%")
        

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


def preparar_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Divide el dataset en entrenamiento y test según FECHA_CORTE y DIAS_TEST.

    Para esta evaluación se usa una ventana de entrenamiento acotada mediante
    TRAIN_DIAS. Esto permite validar SARIMAX sin entrenar sobre todo el histórico.
    """
    inicio_test = pd.Timestamp(FECHA_CORTE)
    fin_test = inicio_test + pd.Timedelta(days=DIAS_TEST)
    inicio_train = inicio_test - pd.Timedelta(days=TRAIN_DIAS)

    train = df[(df.index >= inicio_train) & (df.index < inicio_test)].copy()
    test = df[(df.index >= inicio_test) & (df.index < fin_test)].copy()

    if train.empty:
        raise ValueError(
            "El conjunto de entrenamiento quedó vacío. "
            "Revisa FECHA_CORTE o TRAIN_DIAS."
        )

    if test.empty:
        raise ValueError(
            "El conjunto de test quedó vacío. "
            "Revisa FECHA_CORTE o DIAS_TEST."
        )

    return train, test


def diagnosticar_nulos_detallado(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """
    Diagnóstico detallado de nulos e infinitos para SARIMAX.

    Permite identificar en qué columnas y fechas aparecen problemas
    antes de aplicar cualquier imputación.
    """
    columnas_modelo = COLUMNAS_NECESARIAS

    print("\nDiagnóstico detallado de calidad de datos")
    print("-" * 60)

    for nombre, datos in [("TRAIN", train), ("TEST", test)]:
        print(f"\n{nombre}")

        nulos = datos[columnas_modelo].isna().sum()
        nulos = nulos[nulos > 0]

        if nulos.empty:
            print("No se detectaron nulos.")
        else:
            print("\nNulos por columna:")
            print(nulos)

            filas_con_nulos = datos[datos[columnas_modelo].isna().any(axis=1)]

            print("\nPrimeras fechas con nulos:")
            print(filas_con_nulos[columnas_modelo].head(20))

            print("\nÚltimas fechas con nulos:")
            print(filas_con_nulos[columnas_modelo].tail(20))

        numericas = datos[columnas_modelo].select_dtypes(include=[np.number])
        infinitos = np.isinf(numericas).sum()
        infinitos = infinitos[infinitos > 0]

        if infinitos.empty:
            print("\nNo se detectaron infinitos.")
        else:
            print("\nInfinitos por columna:")
            print(infinitos)


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

        datos_numericos = datos.select_dtypes(include=[np.number]) if isinstance(datos, pd.DataFrame) else datos

        if np.isinf(datos_numericos).sum().sum() > 0:
            raise ValueError(f"{nombre} contiene valores infinitos.")


def preparar_exog(datos: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara variables exógenas para SARIMAX.

    Incluye:
        - Variables meteorológicas.
        - Variables básicas de calendario.
        - Dummies de día de semana.
        - Opcionalmente, demanda_lag_168h como variante experimental.

    Justificación:
        El SARIMAX base sobreestimó sábado y domingo. Las dummies de
        dia_semana ayudan a capturar diferencias estructurales entre días.
        La variable demanda_lag_168h se incorpora como experimento para capturar
        dependencia semanal explícita.
    """
    exog = datos[REGRESORES_BASE].copy()

    if USAR_LAG_168H:
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
) -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Construye las matrices endógenas y exógenas para SARIMAX.

    La preparación conjunta de variables exógenas garantiza que train y test
    tengan exactamente las mismas columnas, especialmente cuando se generan
    variables dummy.
    """
    datos_completos = pd.concat([train, test], axis=0)

    exog_completa = preparar_exog(datos_completos)

    exog_train = exog_completa.loc[train.index]
    exog_test = exog_completa.loc[test.index]

    y_train = train["demanda_mw"].astype(float)
    y_test = test["demanda_mw"].astype(float)

    validar_sin_nulos_para_modelo(y_train, exog_train, y_test, exog_test)

    return y_train, exog_train, y_test, exog_test


def entrenar_sarimax(y_train: pd.Series, exog_train: pd.DataFrame):
    """
    Entrena un modelo SARIMAX con variable objetivo y regresores exógenos.

    Para evitar problemas derivados de índices temporales sin frecuencia
    explícita, statsmodels recibe un índice numérico interno. Las fechas se
    conservan fuera del modelo para evaluación y visualización.
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

    El modelo usa un índice numérico interno, pero la predicción recupera
    después el índice temporal original del conjunto de test.
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


def guardar_grafica(real: pd.Series, estimado: pd.Series) -> None:
    """
    Guarda una gráfica comparando demanda real y predicción SARIMAX.
    """
    plt.figure(figsize=(14, 5))

    plt.plot(real.index, real.values, label="Demanda real")
    plt.plot(estimado.index, estimado.values, label="Predicción SARIMAX")

    plt.title(f"SARIMAX - Corte {FECHA_CORTE} + {DIAS_TEST} días")
    plt.xlabel("Fecha")
    plt.ylabel("MW")
    plt.legend()

    eje = plt.gca()
    eje.xaxis.set_major_locator(mdates.HourLocator(interval=INTERVALO_HORAS_GRAFICA))
    eje.xaxis.set_major_formatter(mdates.DateFormatter("%a %d/%m %H:%M"))

    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.savefig(FICHERO_GRAFICA)
    plt.close()


# =============================================================================
# 4. Ejecución principal
# =============================================================================

def main() -> None:
    """
    Ejecuta la evaluación simple de SARIMAX.
    """
    print("Cargando dataset...")
    df = cargar_dataset(FICHERO)

    print("Preparando train/test...")
    train, test = preparar_train_test(df)

    print(f"Ventana de entrenamiento: últimos {TRAIN_DIAS} días antes de {FECHA_CORTE}")
    print(f"Filas entrenamiento: {len(train)}")
    print(f"Filas test: {len(test)}")
    print(f"Regresores base: {REGRESORES_BASE}")
    print(f"USAR_LAG_168H: {USAR_LAG_168H}")
    print(f"ORDER: {ORDER}")
    print(f"SEASONAL_ORDER: {SEASONAL_ORDER}")

    diagnosticar_nulos_detallado(train, test)

    y_train, exog_train, y_test, exog_test = preparar_matrices_modelo(train, test)

    print(f"Columnas exógenas usadas: {list(exog_train.columns)}")

    print("Entrenando SARIMAX...")
    resultado = entrenar_sarimax(y_train, exog_train)

    print("Generando predicción...")
    estimado = predecir(resultado, exog_test)

    mape = calcular_mape(y_test, estimado)
    guardar_grafica(y_test, estimado)

    nombre_variante = "SARIMAX + lag 168h" if USAR_LAG_168H else "SARIMAX base"

    print(f"\nMAPE {nombre_variante} (corte {FECHA_CORTE}, {DIAS_TEST} días): {mape:.2f}%")

    evaluar_mape_por_dia(
        real=y_test,
        estimado=estimado,
        nombre_modelo=nombre_variante,
    )

    pred_lag_24h, pred_lag_168h = evaluar_baselines_naive(test)

    if pred_lag_24h is not None:
        evaluar_mape_por_dia(
            real=test.loc[pred_lag_24h.index, "demanda_mw"],
            estimado=pred_lag_24h,
            nombre_modelo="Baseline lag 24h",
        )

    if pred_lag_168h is not None:
        evaluar_mape_por_dia(
            real=test.loc[pred_lag_168h.index, "demanda_mw"],
            estimado=pred_lag_168h,
            nombre_modelo="Baseline lag 168h",
        )

    print(f"\nGráfica guardada en: {FICHERO_GRAFICA}")

if __name__ == "__main__":
    main()    
