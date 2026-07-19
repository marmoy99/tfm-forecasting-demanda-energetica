"""
sarimax_final.py

Objetivo:
    Entrenar el modelo SARIMAX final sobre el dataset de demanda eléctrica
    horaria y generar una predicción explícita para los días 23, 24 y 25 de
    marzo de 2026, con granularidad horaria.

Modelo utilizado:
    SARIMAX + demanda_lag_168h.

Justificación metodológica:
    En la evaluación walk-forward, SARIMAX + lag 168h fue la variante SARIMAX
    con mejor MAPE medio frente al SARIMAX base. La variable demanda_lag_168h
    representa la demanda observada en la misma hora de la semana anterior y
    permite capturar dependencia semanal explícita.

Escenario final:
    - Corte de entrenamiento: hasta 2026-03-22 23:00:00.
    - Horizonte de predicción: 2026-03-23 00:00:00 a 2026-03-25 23:00:00.
    - Ventana de entrenamiento: últimos 365 días disponibles antes del horizonte.
    - Meteorología futura: perfil histórico medio por mes y hora.
    - Calendario futuro: generado directamente desde la fecha.
    - demanda_lag_168h futura: tomada de la demanda observada 168 horas antes.

Salidas:
    - reports/predictions/prediccion_sarimax_final.csv
    - reports/figures/sarimax_final.png
"""

import os
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    import holidays
except ImportError:  # pragma: no cover - se valida en tiempo de ejecución.
    holidays = None


# =============================================================================
# 1. Definición de rutas
# =============================================================================

CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))

# Subimos desde src/models/SARIMAX hasta la raíz del repositorio.
RAIZ_REPOSITORIO = os.path.abspath(os.path.join(CARPETA_SCRIPT, "..", "..", ".."))

FICHERO = os.path.join(
    RAIZ_REPOSITORIO,
    "data",
    "processed",
    "dataset_modelado.csv",
)

CARPETA_GRAFICAS = os.path.join(RAIZ_REPOSITORIO, "reports", "figures")
os.makedirs(CARPETA_GRAFICAS, exist_ok=True)

FICHERO_PREDICCION = os.path.join(
    RAIZ_REPOSITORIO,
    "reports",
    "predictions",
    "prediccion_sarimax_final.csv",
)

FICHERO_GRAFICA = os.path.join(
    CARPETA_GRAFICAS,
    "sarimax_final.png",
)


# =============================================================================
# 2. Parámetros del escenario final
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

REGRESORES_METEO = [
    "HDD",
    "CDD",
    "humedad_relativa",
    "velocidad_viento",
    "radiacion_solar",
]

COLUMNAS_NECESARIAS = [
    "demanda_mw",
    "hora",
    "dia_semana",
    "mes",
    "demanda_lag_168h",
] + REGRESORES_BASE

# Escenario explícito acordado para la predicción final.
FECHA_CORTE_FINAL = pd.Timestamp("2026-03-22 00:00:00")
INICIO_PREDICCION = pd.Timestamp("2026-03-23 00:00:00")
DIAS_A_PREDECIR = 3
HORAS_A_PREDECIR = DIAS_A_PREDECIR * 24

# Coherente con la evaluación walk-forward.
TRAIN_DIAS = 365

# Configuración utilizada durante la evaluación SARIMAX.
ORDER = (1, 1, 1)
SEASONAL_ORDER = (1, 0, 0, 24)
MAXITER = 25

# Se mantiene la variante enriquecida, validada como la mejor variante SARIMAX
# en el walk-forward comparativo.
USAR_LAG_168H = True

# Para visualización: días históricos previos que se muestran junto a la predicción.
DIAS_HISTORICO_GRAFICA = 7
INTERVALO_HORAS_GRAFICA = 12


# =============================================================================
# 3. Funciones auxiliares
# =============================================================================

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
    locales pueden existir discontinuidades asociadas al cambio horario. Forzar
    la frecuencia puede crear filas artificiales con NaN.
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


def preparar_train_final(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filtra el histórico final usado para entrenar SARIMAX.

    El entrenamiento incluye los últimos TRAIN_DIAS días disponibles antes del
    horizonte de predicción y termina exactamente en FECHA_CORTE_FINAL.
    """
    if FECHA_CORTE_FINAL not in df.index:
        ultima_fecha = df.index.max()
        raise ValueError(
            "No se encontró la fecha de corte final en el dataset: "
            f"{FECHA_CORTE_FINAL}. Última fecha disponible: {ultima_fecha}."
        )

    inicio_train = INICIO_PREDICCION - pd.Timedelta(days=TRAIN_DIAS)

    train = df[
        (df.index >= inicio_train)
        & (df.index <= FECHA_CORTE_FINAL)
    ].copy()

    if train.empty:
        raise ValueError("El conjunto de entrenamiento final quedó vacío.")

    return train


def construir_indice_futuro() -> pd.DatetimeIndex:
    """
    Construye el índice horario futuro del escenario final.
    """
    futuro_index = pd.date_range(
        start=INICIO_PREDICCION,
        periods=HORAS_A_PREDECIR,
        freq="h",
    )

    fin_esperado = pd.Timestamp("2026-03-25 23:00:00")

    if futuro_index[-1] != fin_esperado:
        raise ValueError(
            "El índice futuro no termina en la fecha esperada. "
            f"Fin obtenido: {futuro_index[-1]}, fin esperado: {fin_esperado}."
        )

    return futuro_index


def generar_calendario_futuro(futuro: pd.DataFrame) -> pd.DataFrame:
    """
    Genera variables de calendario futuras a partir del índice temporal.
    """
    if holidays is None:
        raise ImportError(
            "La librería 'holidays' no está instalada. "
            "Instálala con: python -m pip install holidays"
        )

    futuro = futuro.copy()
    futuro["hora"] = futuro.index.hour
    futuro["dia_semana"] = futuro.index.dayofweek
    futuro["mes"] = futuro.index.month
    futuro["es_fin_de_semana"] = futuro["dia_semana"].isin([5, 6]).astype(int)
    futuro["es_agosto"] = (futuro["mes"] == 8).astype(int)

    calendario_es = holidays.country_holidays("ES", years=sorted(futuro.index.year.unique()))
    futuro["es_festivo"] = futuro.index.normalize().map(
        lambda fecha: int(fecha.date() in calendario_es)
    )

    return futuro


def imputar_meteorologia_futura(
    historico: pd.DataFrame,
    futuro: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estima variables meteorológicas futuras usando perfiles históricos medios.

    Perfil principal:
        media histórica por mes y hora.

    Respaldos:
        1. media histórica por hora;
        2. media global de la variable.

    Esta estrategia evita repetir el último valor observado y preserva mejor la
    estructura horaria y estacional de las variables meteorológicas.
    """
    futuro = futuro.copy()
    historico = historico.copy()

    if "mes" not in historico.columns:
        historico["mes"] = historico.index.month
    if "hora" not in historico.columns:
        historico["hora"] = historico.index.hour

    for variable in REGRESORES_METEO:
        perfil_mes_hora = historico.groupby(["mes", "hora"])[variable].mean()
        perfil_hora = historico.groupby("hora")[variable].mean()
        media_global = historico[variable].mean()

        valores = []
        for fecha, fila in futuro.iterrows():
            clave_mes_hora = (fila["mes"], fila["hora"])

            if clave_mes_hora in perfil_mes_hora.index:
                valor = perfil_mes_hora.loc[clave_mes_hora]
            elif fila["hora"] in perfil_hora.index:
                valor = perfil_hora.loc[fila["hora"]]
            else:
                valor = media_global

            valores.append(valor)

        futuro[variable] = valores

    return futuro


def agregar_lag_168h_futuro(
    historico: pd.DataFrame,
    futuro: pd.DataFrame,
) -> pd.DataFrame:
    """
    Agrega demanda_lag_168h al dataframe futuro.

    Para cada hora futura se busca la demanda observada exactamente 168 horas
    antes. Si algún timestamp no está disponible, se detiene la ejecución para
    evitar una predicción apoyada en datos incompletos.
    """
    futuro = futuro.copy()
    valores_lag = []
    faltantes = []

    for fecha in futuro.index:
        fecha_lag = fecha - pd.Timedelta(hours=168)

        if fecha_lag not in historico.index:
            faltantes.append(fecha_lag)
            valores_lag.append(np.nan)
        else:
            valores_lag.append(historico.loc[fecha_lag, "demanda_mw"])

    futuro["demanda_lag_168h"] = valores_lag

    if faltantes:
        muestra = ", ".join(str(fecha) for fecha in faltantes[:5])
        raise ValueError(
            "No se pudieron construir todos los valores de demanda_lag_168h. "
            f"Primeros timestamps faltantes: {muestra}"
        )

    return futuro


def construir_futuro(df_hasta_corte: pd.DataFrame) -> pd.DataFrame:
    """
    Construye el dataframe futuro completo para SARIMAX final.
    """
    futuro = pd.DataFrame(index=construir_indice_futuro())

    futuro = generar_calendario_futuro(futuro)
    futuro = imputar_meteorologia_futura(
        historico=df_hasta_corte,
        futuro=futuro,
    )
    futuro = agregar_lag_168h_futuro(
        historico=df_hasta_corte,
        futuro=futuro,
    )

    return futuro


def diagnosticar_festivos_final(
    df_hasta_corte: pd.DataFrame,
    futuro: pd.DataFrame,
) -> None:
    """
    Compara presencia de festivos en el horizonte predicho y en su semana lag.
    """
    fechas_test = sorted({
        fecha.strftime("%Y-%m-%d")
        for fecha in futuro[futuro["es_festivo"] == 1].index.date
    })

    indices_lag = futuro.index - pd.Timedelta(hours=168)
    lag_periodo = df_hasta_corte.loc[df_hasta_corte.index.intersection(indices_lag)].copy()

    if "es_festivo" in lag_periodo.columns and not lag_periodo.empty:
        fechas_lag = sorted({
            fecha.strftime("%Y-%m-%d")
            for fecha in lag_periodo[lag_periodo["es_festivo"] == 1].index.date
        })
    else:
        fechas_lag = []

    print(
        "Festivos horizonte predicho: "
        f"{';'.join(fechas_test) if fechas_test else 'sin festivos'} | "
        "Festivos semana lag 168h: "
        f"{';'.join(fechas_lag) if fechas_lag else 'sin festivos'}"
    )


def validar_sin_nulos_para_modelo(
    y_train: pd.Series,
    exog_train: pd.DataFrame,
    exog_futuro: pd.DataFrame,
) -> None:
    """
    Detiene la ejecución si existen nulos o infinitos en las matrices del modelo.
    """
    objetos = {
        "y_train": y_train,
        "exog_train": exog_train,
        "exog_futuro": exog_futuro,
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


def preparar_exog(datos: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara variables exógenas para SARIMAX.

    Incluye:
        - Variables meteorológicas estimadas u observadas.
        - Variables básicas de calendario.
        - demanda_lag_168h.
        - Dummies de día de semana.
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
    futuro: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Construye y_train, exog_train y exog_futuro.

    La preparación conjunta de train y futuro garantiza que las dummies de
    día de semana tengan exactamente las mismas columnas.
    """
    datos_completos = pd.concat([train, futuro], axis=0)

    exog_completa = preparar_exog(datos_completos)

    exog_train = exog_completa.loc[train.index]
    exog_futuro = exog_completa.loc[futuro.index]

    y_train = train["demanda_mw"].astype(float)

    validar_sin_nulos_para_modelo(y_train, exog_train, exog_futuro)

    return y_train, exog_train, exog_futuro


def entrenar_sarimax(y_train: pd.Series, exog_train: pd.DataFrame):
    """
    Entrena SARIMAX con índice numérico interno.

    Las fechas se conservan fuera del modelo para exportación y visualización,
    pero statsmodels recibe índices consecutivos para evitar problemas con
    índices temporales locales sin frecuencia explícita.
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


def predecir(resultado, exog_futuro: pd.DataFrame) -> pd.Series:
    """
    Genera la predicción SARIMAX para el horizonte futuro.
    """
    indice_temporal = exog_futuro.index

    exog_modelo = exog_futuro.copy()
    exog_modelo.index = pd.RangeIndex(start=0, stop=len(exog_modelo))

    pred = resultado.get_forecast(
        steps=len(exog_modelo),
        exog=exog_modelo,
    )

    estimado = pred.predicted_mean
    estimado.index = indice_temporal

    return estimado


def guardar_prediccion(
    prediccion: pd.Series,
    futuro: pd.DataFrame,
) -> None:
    """
    Guarda la predicción final y las variables futuras utilizadas.
    """
    salida = futuro.copy()
    salida["prediccion_demanda_mw"] = prediccion
    salida["modelo"] = "SARIMAX + lag 168h"
    salida["fecha_corte_entrenamiento"] = FECHA_CORTE_FINAL
    salida["train_dias"] = TRAIN_DIAS
    salida["order"] = str(ORDER)
    salida["seasonal_order"] = str(SEASONAL_ORDER)
    salida["meteo_futura_metodo"] = "perfil_historico_medio_mes_hora"

    salida = salida.reset_index().rename(columns={"index": "datetime"})

    salida.to_csv(FICHERO_PREDICCION, index=False, encoding="utf-8-sig")


def guardar_grafica(
    historico: pd.DataFrame,
    prediccion: pd.Series,
) -> None:
    """
    Guarda una gráfica con histórico reciente y predicción SARIMAX final.
    """
    inicio_grafica = INICIO_PREDICCION - pd.Timedelta(days=DIAS_HISTORICO_GRAFICA)
    historico_grafica = historico[historico.index >= inicio_grafica]

    plt.figure(figsize=(14, 5))

    plt.plot(
        historico_grafica.index,
        historico_grafica["demanda_mw"],
        label="Demanda observada",
    )
    plt.plot(
        prediccion.index,
        prediccion.values,
        label="Predicción SARIMAX final",
    )

    plt.axvline(FECHA_CORTE_FINAL, linestyle="--", label="Corte entrenamiento")

    plt.title("Predicción final SARIMAX - 23, 24 y 25 de marzo de 2026")
    plt.xlabel("Fecha")
    plt.ylabel("MW")
    plt.legend()

    eje = plt.gca()
    eje.xaxis.set_major_locator(mdates.HourLocator(interval=INTERVALO_HORAS_GRAFICA))
    eje.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m %H:%M"))

    plt.xticks(rotation=35)
    plt.tight_layout()
    plt.savefig(FICHERO_GRAFICA)
    plt.close()


# =============================================================================
# 4. Ejecución principal
# =============================================================================

def main() -> None:
    """
    Ejecuta la predicción final SARIMAX.
    """
    print("Cargando dataset...")
    df = cargar_dataset(FICHERO)

    print(f"Última fecha disponible en dataset: {df.index.max()}")
    print(f"Corte final definido: {FECHA_CORTE_FINAL}")
    print(f"Horizonte final: {INICIO_PREDICCION} a {INICIO_PREDICCION + pd.Timedelta(hours=HORAS_A_PREDECIR - 1)}")

    df_hasta_corte = df[df.index <= FECHA_CORTE_FINAL].copy()

    print("Preparando ventana de entrenamiento final...")
    train = preparar_train_final(df)

    print(f"Filas train: {len(train)}")
    print(f"Inicio train: {train.index.min()}")
    print(f"Fin train: {train.index.max()}")
    print(f"ORDER: {ORDER}")
    print(f"SEASONAL_ORDER: {SEASONAL_ORDER}")

    print("Construyendo variables futuras...")
    futuro = construir_futuro(df_hasta_corte)
    diagnosticar_festivos_final(df_hasta_corte=df_hasta_corte, futuro=futuro)

    y_train, exog_train, exog_futuro = preparar_matrices_modelo(
        train=train,
        futuro=futuro,
    )

    print(f"Columnas exógenas usadas: {list(exog_train.columns)}")

    print("Entrenando SARIMAX final...")
    resultado = entrenar_sarimax(y_train=y_train, exog_train=exog_train)

    print("Generando predicción final...")
    prediccion = predecir(resultado=resultado, exog_futuro=exog_futuro)

    guardar_prediccion(prediccion=prediccion, futuro=futuro)
    guardar_grafica(historico=df_hasta_corte, prediccion=prediccion)

    print("\nPredicción final generada correctamente.")
    print(f"CSV guardado en: {FICHERO_PREDICCION}")
    print(f"Gráfica guardada en: {FICHERO_GRAFICA}")
    print("\nPrimeras filas de la predicción:")
    print(prediccion.head().round(2).to_string())


if __name__ == "__main__":
    main()
