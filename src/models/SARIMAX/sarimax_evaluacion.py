"""
sarimax_evaluacion.py

Objetivo:
    Entrenar y evaluar un primer modelo SARIMAX sobre el dataset de demanda
    eléctrica horaria del TFM, usando un corte temporal simple comparable con
    el script prophet_evaluacion.py.

Salida:
    - MAPE del periodo de test.
    - Gráfico comparativo demanda real vs predicción SARIMAX.

Nota metodológica:
    En esta primera versión no se usan demanda_lag_24h ni demanda_lag_168h
    como variables exógenas para evitar riesgo de leakage en predicción futura.
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
REGRESORES = [
    "HDD",
    "CDD",
    "humedad_relativa",
    "velocidad_viento",
    "radiacion_solar",
    "es_festivo",
    "es_fin_de_semana",
    "es_agosto",
]

FECHA_CORTE = "2025-10-03"
DIAS_TEST = 3
INTERVALO_HORAS_GRAFICA = 6

# Configuración inicial simple para datos horarios.
# order=(p,d,q): componente no estacional.
# seasonal_order=(P,D,Q,s): componente estacional con ciclo diario de 24 horas.
ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 0, 1, 24)
MAXITER = 50


# =============================================================================
# 3. Funciones auxiliares
# =============================================================================

def calcular_mape(real: pd.Series, estimado: pd.Series) -> float:
    """Calcula el MAPE en porcentaje."""
    return (np.abs(real - estimado) / real).mean() * 100


def validar_columnas(df: pd.DataFrame, columnas_necesarias: list[str]) -> None:
    """Comprueba que el dataset contiene todas las columnas necesarias."""
    columnas_faltantes = [col for col in columnas_necesarias if col not in df.columns]
    if columnas_faltantes:
        raise ValueError(f"Faltan columnas en el dataset: {columnas_faltantes}")


def cargar_dataset(ruta: str) -> pd.DataFrame:
    """Carga el dataset, ordena por fecha y fija frecuencia horaria."""
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el dataset en la ruta: {ruta}")

    df = pd.read_csv(ruta, parse_dates=["datetime"])
    df = df.sort_values("datetime").drop_duplicates(subset="datetime")
    df = df.set_index("datetime")

    # SARIMAX funciona mejor si el índice temporal tiene frecuencia explícita.
    # Si falta alguna hora, asfreq('h') la hará visible como NaN.
    df = df.asfreq("h")

    columnas_necesarias = ["demanda_mw"] + REGRESORES
    validar_columnas(df.reset_index(), ["datetime"] + columnas_necesarias)

    # En esta versión simple eliminamos filas con nulos en target o regresores.
    # El dataset modelado no debería tener nulos en estas variables.
    df = df.dropna(subset=columnas_necesarias)

    return df


def preparar_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide el dataset en entrenamiento y test según FECHA_CORTE y DIAS_TEST."""
    inicio_test = pd.Timestamp(FECHA_CORTE)
    fin_test = inicio_test + pd.Timedelta(days=DIAS_TEST)

    train = df[df.index < inicio_test].copy()
    test = df[(df.index >= inicio_test) & (df.index < fin_test)].copy()

    if train.empty:
        raise ValueError("El conjunto de entrenamiento quedó vacío. Revisa FECHA_CORTE.")
    if test.empty:
        raise ValueError("El conjunto de test quedó vacío. Revisa FECHA_CORTE y DIAS_TEST.")

    return train, test


def entrenar_sarimax(train: pd.DataFrame):
    """Entrena el modelo SARIMAX con demanda_mw como target y regresores externos."""
    y_train = train["demanda_mw"]
    exog_train = train[REGRESORES]

    modelo = SARIMAX(
        endog=y_train,
        exog=exog_train,
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resultado = modelo.fit(disp=False, maxiter=MAXITER)

    return resultado


def predecir(resultado, train: pd.DataFrame, test: pd.DataFrame) -> pd.Series:
    """Genera predicciones para el periodo de test."""
    inicio = len(train)
    fin = len(train) + len(test) - 1

    prediccion = resultado.predict(
        start=inicio,
        end=fin,
        exog=test[REGRESORES],
    )

    prediccion.index = test.index
    return prediccion


def guardar_grafica(real: pd.Series, estimado: pd.Series) -> None:
    """Guarda una gráfica comparativa de demanda real vs predicción."""
    plt.figure(figsize=(14, 5))
    plt.plot(real, label="Demanda real", linewidth=2)
    plt.plot(estimado, label="Predicción SARIMAX", linewidth=2)
    plt.legend()
    plt.ylabel("MW")
    plt.title(f"SARIMAX - Corte {FECHA_CORTE} + {DIAS_TEST} días")

    minimo = min(real.min(), estimado.min())
    maximo = max(real.max(), estimado.max())
    limite_inferior = int(minimo // 1000) * 1000
    limite_superior = int(maximo // 1000) * 1000 + 1000

    plt.ylim(limite_inferior, limite_superior)
    plt.yticks(np.arange(limite_inferior, limite_superior + 1000, 1000))

    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=INTERVALO_HORAS_GRAFICA))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%a %d/%m %H:%M"))
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    plt.savefig(FICHERO_GRAFICA, dpi=110)
    plt.close()


# =============================================================================
# 4. Ejecución principal
# =============================================================================

def main() -> None:
    """Ejecuta la evaluación simple de SARIMAX."""
    print("Cargando dataset...")
    df = cargar_dataset(FICHERO)

    print("Preparando train/test...")
    train, test = preparar_train_test(df)

    print(f"Filas entrenamiento: {len(train)}")
    print(f"Filas test: {len(test)}")
    print(f"Regresores: {REGRESORES}")
    print(f"ORDER: {ORDER}")
    print(f"SEASONAL_ORDER: {SEASONAL_ORDER}")

    print("Entrenando SARIMAX...")
    resultado = entrenar_sarimax(train)

    print("Generando predicción...")
    estimado = predecir(resultado, train, test)
    real = test["demanda_mw"]

    mape = calcular_mape(real, estimado)
    guardar_grafica(real, estimado)

    print(f"MAPE SARIMAX (corte {FECHA_CORTE}, {DIAS_TEST} días): {mape:.2f}%")
    print(f"Gráfica guardada en: {FICHERO_GRAFICA}")


if __name__ == "__main__":
    main()
