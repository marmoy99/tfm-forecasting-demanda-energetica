"""
sarimax_armonico_final.py

Objetivo
--------
Entrenar el modelo SARIMAX armónico + lag 168h seleccionado como candidato
final y generar:

1. Un CSV con los resultados walk-forward del modelo seleccionado:
   corte, modelo, mape, mae, rmse

2. Un CSV con la predicción horaria final:
   datetime
   prediccion_demanda_mw
   modelo
   fecha_corte_entrenamiento
   demanda_lag_168h
   HDD
   CDD
   humedad_relativa
   velocidad_viento
   radiacion_solar
   es_festivo

Modelo
------
- order=(2, 0, 2)
- seasonal_order=(0, 0, 0, 0)
- trend="c"
- Fourier:
    diaria:  periodo 24, K=6
    semanal: periodo 168, K=3
    anual:   periodo 8766, K=3
- demanda_lag_168h
- meteorología
- calendario
- estandarización calculada exclusivamente con el train

Nota
----
Las métricas exportadas proceden del walk-forward comparativo. No son métricas
de la predicción futura, porque para esta última todavía no existe demanda real.
"""

from __future__ import annotations

import os
import warnings

import holidays
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
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

FICHERO_COMPARACION = os.path.join(
    RAIZ_REPOSITORIO,
    "reports",
    "model_comparison",
    "resultados_sarimax_comparacion.csv",
)

CARPETA_RESULTADOS = os.path.join(
    RAIZ_REPOSITORIO,
    "reports",
    "model_comparison",
)

CARPETA_PREDICCIONES = os.path.join(
    RAIZ_REPOSITORIO,
    "reports",
    "predictions",
)

FICHERO_METRICAS = os.path.join(
    CARPETA_RESULTADOS,
    "resultados_sarimax_armonico_final.csv",
)

FICHERO_PREDICCION = os.path.join(
    CARPETA_PREDICCIONES,
    "prediccion_sarimax_armonico_final.csv",
)

CARPETA_GRAFICAS = os.path.join(
    RAIZ_REPOSITORIO,
    "reports",
    "figures",
)

FICHERO_GRAFICA = os.path.join(
    CARPETA_GRAFICAS,
    "sarimax_armonico_final.png",
)


# =============================================================================
# 2. Configuración del modelo final
# =============================================================================

NOMBRE_MODELO = "SARIMAX armónico + lag 168h"

REGRESORES_METEO = [
    "HDD",
    "CDD",
    "humedad_relativa",
    "velocidad_viento",
    "radiacion_solar",
]

ORDER = (2, 0, 2)
SEASONAL_ORDER = (0, 0, 0, 0)
TREND = "c"
MAXITER = 300

FOURIER = [
    (24, 6),
    (168, 3),
    (8766, 3),
]

DIAS_TRAIN = 365
HORAS_TRAIN = DIAS_TRAIN * 24
HORAS_PREDICCION = 72
DIAS_HISTORICO_GRAFICA = 7
INTERVALO_HORAS_GRAFICA = 12


# =============================================================================
# 3. Preparación del dataset
# =============================================================================

def validar_columnas(df: pd.DataFrame, columnas: list[str]) -> None:
    """Comprueba que el dataset contiene las columnas requeridas."""
    faltantes = [col for col in columnas if col not in df.columns]

    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {faltantes}")


def cargar_y_preparar_dataset(ruta: str) -> pd.DataFrame:
    """
    Aplica la misma preparación utilizada en el walk-forward comparativo:

    - índice horario continuo;
    - interpolación temporal de demanda y meteorología;
    - calendario regenerado desde el índice;
    - demanda_lag_168h calculada tras regularizar la serie.
    """
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el dataset: {ruta}")

    df = pd.read_csv(ruta, parse_dates=["datetime"])

    # El estudio termina el 31-mar-2026: abril-2026 es dato provisional de Esios
    # (demanda ~40% por debajo de lo real) y quedo excluido por decision del grupo.
    df = df[df["datetime"] <= "2026-03-31 23:00:00"]

    validar_columnas(
        df,
        ["datetime", "demanda_mw"] + REGRESORES_METEO,
    )

    df = df.set_index("datetime").sort_index()

    if df.index.has_duplicates:
        raise ValueError(
            f"El índice contiene {df.index.duplicated().sum()} "
            "timestamps duplicados."
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
        years=range(df.index.min().year, df.index.max().year + 2)
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
        raise ValueError(
            "Persisten nulos o infinitos tras la preparación:\n"
            f"{nulos}"
        )

    print(
        f"Dataset preparado: {len(df):,} horas | "
        f"huecos interpolados: {huecos}"
    )

    return df


# =============================================================================
# 4. Ventana de entrenamiento y horizonte final
# =============================================================================

def preparar_train_final(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selecciona las últimas 8.760 observaciones disponibles.

    El corte final se determina a partir del último dato observado real del
    dataset. No se fabrican horas posteriores al último registro disponible.
    """
    if len(df) < HORAS_TRAIN:
        raise ValueError(
            f"No hay suficientes observaciones para entrenar {DIAS_TRAIN} días."
        )

    train = df.iloc[-HORAS_TRAIN:].copy()

    if len(train) != HORAS_TRAIN:
        raise ValueError(
            f"Train final incompleto: {len(train)} filas."
        )

    return train


def construir_horizonte_completo(
    fecha_corte: pd.Timestamp,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """
    Construye dos índices:

    - indice_forecast:
      todas las horas desde la hora posterior al corte hasta el final del
      horizonte objetivo;

    - indice_objetivo:
      las 72 horas correspondientes a los tres días completos siguientes al
      día del corte.

    Esta lógica permite mantener como horizonte final días completos incluso si
    el dataset termina a las 00:00. Las horas intermedias se pronostican, no se
    imputan como observaciones.
    """
    inicio_objetivo = fecha_corte.normalize() + pd.Timedelta(days=1)

    indice_objetivo = pd.date_range(
        start=inicio_objetivo,
        periods=HORAS_PREDICCION,
        freq="h",
    )

    indice_forecast = pd.date_range(
        start=fecha_corte + pd.Timedelta(hours=1),
        end=indice_objetivo[-1],
        freq="h",
    )

    return indice_forecast, indice_objetivo


# =============================================================================
# 5. Construcción de variables futuras
# =============================================================================

def construir_calendario_futuro(
    indice: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Genera las variables futuras de calendario desde el índice."""
    futuro = pd.DataFrame(index=indice)

    calendario_es = holidays.Spain(
        years=range(indice.min().year, indice.max().year + 1)
    )

    futuro["mes"] = futuro.index.month
    futuro["hora"] = futuro.index.hour
    futuro["dia_semana"] = futuro.index.dayofweek
    futuro["es_fin_de_semana"] = (
        futuro["dia_semana"].isin([5, 6]).astype(int)
    )
    futuro["es_agosto"] = (futuro["mes"] == 8).astype(int)
    futuro["es_festivo"] = futuro.index.map(
        lambda fecha: int(fecha.date() in calendario_es)
    )

    return futuro


def imputar_meteorologia_futura(
    train: pd.DataFrame,
    futuro: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estima la meteorología futura con perfiles históricos del train:

    1. media por mes y hora;
    2. respaldo por hora;
    3. respaldo global.

    La metodología debe describirse como meteorología estimada, no como
    predicción meteorológica operativa.
    """
    futuro = futuro.copy()

    for variable in REGRESORES_METEO:
        perfil_mes_hora = train.groupby(["mes", "hora"])[variable].mean()
        perfil_hora = train.groupby("hora")[variable].mean()
        media_global = train[variable].mean()

        valores = []

        for mes, hora in zip(futuro["mes"], futuro["hora"]):
            clave = (mes, hora)

            if clave in perfil_mes_hora.index:
                valor = perfil_mes_hora.loc[clave]
            elif hora in perfil_hora.index:
                valor = perfil_hora.loc[hora]
            else:
                valor = media_global

            valores.append(float(valor))

        futuro[variable] = valores

    return futuro


def agregar_lag_168h_futuro(
    historico: pd.DataFrame,
    futuro: pd.DataFrame,
) -> pd.DataFrame:
    """
    Agrega la demanda observada exactamente 168 horas antes.

    Para un horizonte inferior a siete días, todas las referencias pertenecen
    al histórico observado y no requieren predicciones recursivas del lag.
    """
    futuro = futuro.copy()
    fechas_lag = futuro.index - pd.Timedelta(hours=168)

    faltantes = fechas_lag.difference(historico.index)

    if len(faltantes) > 0:
        muestra = ", ".join(str(fecha) for fecha in faltantes[:5])
        raise ValueError(
            "No se pudo construir demanda_lag_168h para todo el horizonte. "
            f"Primeras fechas faltantes: {muestra}"
        )

    futuro["demanda_lag_168h"] = historico.loc[
        fechas_lag,
        "demanda_mw",
    ].to_numpy()

    return futuro


def construir_futuro(
    historico: pd.DataFrame,
    train: pd.DataFrame,
    indice_forecast: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Construye el dataframe futuro completo."""
    futuro = construir_calendario_futuro(indice_forecast)

    futuro = imputar_meteorologia_futura(
        train=train,
        futuro=futuro,
    )

    futuro = agregar_lag_168h_futuro(
        historico=historico,
        futuro=futuro,
    )

    return futuro


# =============================================================================
# 6. Fourier y matrices del modelo
# =============================================================================

def construir_fourier(
    indice: pd.DatetimeIndex,
    origen: pd.Timestamp,
) -> pd.DataFrame:
    """Construye Fourier con la misma configuración del walk-forward."""
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
    origen_fourier: pd.Timestamp,
) -> pd.DataFrame:
    """
    Construye las exógenas del modelo final:

    - meteorología;
    - calendario;
    - dummies de día de semana;
    - Fourier;
    - demanda_lag_168h.
    """
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

    exog["demanda_lag_168h"] = datos["demanda_lag_168h"]

    return exog.astype(float)


def preparar_matrices_finales(
    train: pd.DataFrame,
    futuro: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Prepara y estandariza las matrices usando únicamente estadísticas del train.
    """
    datos_completos = pd.concat([train, futuro], axis=0)

    exog_completa = construir_exogenas(
        datos=datos_completos,
        origen_fourier=train.index.min(),
    )

    exog_train = exog_completa.loc[train.index]
    exog_futuro = exog_completa.loc[futuro.index]

    media = exog_train.mean()
    desviacion = exog_train.std().replace(0, 1)

    exog_train = (exog_train - media) / desviacion
    exog_futuro = (exog_futuro - media) / desviacion

    y_train = train["demanda_mw"].astype(float)

    for nombre, objeto in {
        "y_train": y_train,
        "exog_train": exog_train,
        "exog_futuro": exog_futuro,
    }.items():
        if objeto.replace([np.inf, -np.inf], np.nan).isna().sum().sum() > 0:
            raise ValueError(f"{nombre} contiene nulos o infinitos.")

    return y_train, exog_train, exog_futuro


# =============================================================================
# 7. Entrenamiento y predicción
# =============================================================================

def entrenar_modelo(
    y_train: pd.Series,
    exog_train: pd.DataFrame,
):
    """Entrena el SARIMAX armónico final."""
    y_modelo = y_train.reset_index(drop=True)
    exog_modelo = exog_train.reset_index(drop=True)

    modelo = SARIMAX(
        endog=y_modelo,
        exog=exog_modelo,
        order=ORDER,
        seasonal_order=SEASONAL_ORDER,
        trend=TREND,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        resultado = modelo.fit(
            disp=False,
            maxiter=MAXITER,
        )

    converge = bool(
        resultado.mle_retvals.get("converged", False)
    )

    if not converge:
        raise RuntimeError(
            "El modelo final no convergió. No se exportará la predicción."
        )

    return resultado


def generar_prediccion(
    resultado,
    exog_futuro: pd.DataFrame,
) -> pd.Series:
    """Genera la predicción para todo el horizonte requerido."""
    exog_modelo = exog_futuro.reset_index(drop=True)

    prediccion = resultado.get_forecast(
        steps=len(exog_modelo),
        exog=exog_modelo,
    ).predicted_mean

    prediccion.index = exog_futuro.index

    return prediccion


# =============================================================================
# 8. Exportación
# =============================================================================

def exportar_metricas_walk_forward() -> pd.DataFrame:
    """
    Filtra del CSV comparativo las 12 ventanas del modelo seleccionado.

    Estas métricas documentan el rendimiento validado del candidato final.
    """
    if not os.path.exists(FICHERO_COMPARACION):
        raise FileNotFoundError(
            "No se encontró el CSV comparativo: "
            f"{FICHERO_COMPARACION}"
        )

    resultados = pd.read_csv(FICHERO_COMPARACION)

    columnas = ["corte", "modelo", "mape", "mae", "rmse"]
    validar_columnas(resultados, columnas)

    seleccion = resultados[
        resultados["modelo"] == NOMBRE_MODELO
    ][columnas].copy()

    if seleccion.empty:
        raise ValueError(
            f"No se encontraron resultados para '{NOMBRE_MODELO}'."
        )

    seleccion = seleccion.sort_values("corte").reset_index(drop=True)

    seleccion.to_csv(
        FICHERO_METRICAS,
        index=False,
        encoding="utf-8-sig",
    )

    return seleccion


def exportar_prediccion_final(
    prediccion_completa: pd.Series,
    futuro_completo: pd.DataFrame,
    indice_objetivo: pd.DatetimeIndex,
    fecha_corte: pd.Timestamp,
) -> pd.DataFrame:
    """
    Exporta las 72 horas del horizonte objetivo con una estructura
    consistente con los CSV finales de LightGBM y Prophet.

    Se conserva demanda_lag_168h como columna adicional de trazabilidad
    específica del modelo SARIMAX.
    """
    prediccion = prediccion_completa.loc[indice_objetivo]
    futuro = futuro_completo.loc[indice_objetivo]

    salida = pd.DataFrame(
        {
            "datetime": indice_objetivo,
            "mes": futuro["mes"].to_numpy(),
            "dia_semana": futuro["dia_semana"].to_numpy(),
            "hora": futuro["hora"].to_numpy(),
            "es_festivo": futuro["es_festivo"].to_numpy(),
            "HDD": futuro["HDD"].to_numpy(),
            "CDD": futuro["CDD"].to_numpy(),
            "humedad_relativa": futuro["humedad_relativa"].to_numpy(),
            "velocidad_viento": futuro["velocidad_viento"].to_numpy(),
            "radiacion_solar": futuro["radiacion_solar"].to_numpy(),
            "prediccion_demanda_mw": prediccion.to_numpy(),
            "modelo": NOMBRE_MODELO,
            "fecha_corte_entrenamiento": fecha_corte,
            "lags": 168,
            "meteo_futura_metodo": "perfil_historico_medio_mes_hora",
            "demanda_lag_168h": futuro["demanda_lag_168h"].to_numpy(),
        }
    )

    columnas_ordenadas = [
        "datetime",
        "mes",
        "dia_semana",
        "hora",
        "es_festivo",
        "HDD",
        "CDD",
        "humedad_relativa",
        "velocidad_viento",
        "radiacion_solar",
        "prediccion_demanda_mw",
        "modelo",
        "fecha_corte_entrenamiento",
        "lags",
        "meteo_futura_metodo",
        "demanda_lag_168h",
    ]

    salida = salida[columnas_ordenadas]

    salida.to_csv(
        FICHERO_PREDICCION,
        index=False,
        encoding="utf-8-sig",
    )

    return salida



def guardar_grafica_final(
    historico: pd.DataFrame,
    prediccion_completa: pd.Series,
    indice_objetivo: pd.DatetimeIndex,
    fecha_corte: pd.Timestamp,
) -> None:
    """
    Guarda una figura con los últimos días observados y la predicción final.

    La gráfica muestra:
    - demanda observada de los últimos DIAS_HISTORICO_GRAFICA días;
    - predicción correspondiente a las 72 horas objetivo;
    - línea vertical en el corte de entrenamiento.
    """
    os.makedirs(CARPETA_GRAFICAS, exist_ok=True)

    inicio_historico = fecha_corte - pd.Timedelta(
        days=DIAS_HISTORICO_GRAFICA
    )

    historico_reciente = historico.loc[
        (historico.index >= inicio_historico)
        & (historico.index <= fecha_corte),
        "demanda_mw",
    ]

    prediccion_objetivo = prediccion_completa.loc[indice_objetivo]

    plt.figure(figsize=(14, 5))

    plt.plot(
        historico_reciente.index,
        historico_reciente.values,
        label="Demanda observada",
        linewidth=2,
    )

    plt.plot(
        prediccion_objetivo.index,
        prediccion_objetivo.values,
        label="Predicción SARIMAX armónico + lag 168h",
        linewidth=2,
    )

    plt.axvline(
        fecha_corte,
        linestyle="--",
        label="Corte de entrenamiento",
    )

    plt.title(
        "Predicción final SARIMAX armónico + lag 168h"
    )
    plt.xlabel("Fecha")
    plt.ylabel("Demanda eléctrica (MW)")
    plt.legend()

    eje = plt.gca()
    eje.xaxis.set_major_locator(
        mdates.HourLocator(interval=INTERVALO_HORAS_GRAFICA)
    )
    eje.xaxis.set_major_formatter(
        mdates.DateFormatter("%d/%m %H:%M")
    )

    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.savefig(FICHERO_GRAFICA, dpi=150)
    plt.close()


# =============================================================================
# 9. Ejecución
# =============================================================================

def main() -> None:
    """Ejecuta el entrenamiento y la predicción final."""
    os.makedirs(CARPETA_RESULTADOS, exist_ok=True)
    os.makedirs(CARPETA_PREDICCIONES, exist_ok=True)
    os.makedirs(CARPETA_GRAFICAS, exist_ok=True)

    print("Cargando y preparando dataset...")
    df = cargar_y_preparar_dataset(FICHERO_DATOS)

    train = preparar_train_final(df)
    fecha_corte = train.index.max()

    indice_forecast, indice_objetivo = construir_horizonte_completo(
        fecha_corte=fecha_corte,
    )

    print("\nEscenario final")
    print("-" * 80)
    print(f"Modelo: {NOMBRE_MODELO}")
    print(f"Train: {train.index.min()} -> {train.index.max()}")
    print(f"Filas train: {len(train)}")
    print(f"Horizonte objetivo: {indice_objetivo[0]} -> {indice_objetivo[-1]}")
    print(f"Horas pronosticadas internamente: {len(indice_forecast)}")
    print(f"Horas exportadas: {len(indice_objetivo)}")
    print(f"ORDER: {ORDER}")
    print(f"FOURIER: {FOURIER}")

    futuro = construir_futuro(
        historico=df,
        train=train,
        indice_forecast=indice_forecast,
    )

    y_train, exog_train, exog_futuro = preparar_matrices_finales(
        train=train,
        futuro=futuro,
    )

    print("\nEntrenando SARIMAX armónico final...")
    resultado = entrenar_modelo(
        y_train=y_train,
        exog_train=exog_train,
    )

    print("Generando predicción...")
    prediccion_completa = generar_prediccion(
        resultado=resultado,
        exog_futuro=exog_futuro,
    )

    metricas = exportar_metricas_walk_forward()

    prediccion = exportar_prediccion_final(
        prediccion_completa=prediccion_completa,
        futuro_completo=futuro,
        indice_objetivo=indice_objetivo,
        fecha_corte=fecha_corte,
    )

    guardar_grafica_final(
        historico=df,
        prediccion_completa=prediccion_completa,
        indice_objetivo=indice_objetivo,
        fecha_corte=fecha_corte,
    )

    print("\nProceso finalizado.")
    print(f"CSV métricas: {FICHERO_METRICAS}")
    print(f"Filas métricas: {len(metricas)}")
    print(f"CSV predicción: {FICHERO_PREDICCION}")
    print(f"Filas predicción: {len(prediccion)}")
    print(f"Gráfica: {FICHERO_GRAFICA}")


if __name__ == "__main__":
    main()
