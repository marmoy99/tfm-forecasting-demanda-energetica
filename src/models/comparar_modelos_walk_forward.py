"""
comparar_modelos_walk_forward.py

Objetivo:
    Construir tablas comparativas de evaluación walk-forward entre Prophet,
    SARIMAX base, SARIMAX + lag 168h y baselines ingenuos.

Entradas esperadas:
    - src/models/SARIMAX/resultados_sarimax_walk_forward.csv
    - src/models/Prophet/resultados_prophet_walk_forward_ventana365.csv

Salidas:
    - reports/model_comparison_walk_forward_by_cut.csv
    - reports/model_comparison_walk_forward_summary.csv

Nota metodológica:
    SARIMAX base se interpreta como modelo estadístico sin rezagos explícitos.
    SARIMAX + lag 168h se interpreta como variante enriquecida con dependencia
    semanal. Los baselines ingenuos se mantienen como referencias de contexto,
    no necesariamente como modelos candidatos finales.
"""

import os

import pandas as pd


# =============================================================================
# 1. Definición de rutas
# =============================================================================

CARPETA_SCRIPT = os.path.dirname(os.path.abspath(__file__))

# Este archivo se espera en src/models/, por lo tanto subimos dos niveles hasta
# la raíz del repositorio.
RAIZ_REPOSITORIO = os.path.abspath(os.path.join(CARPETA_SCRIPT, "..", ".."))

FICHERO_SARIMAX = os.path.join(
    RAIZ_REPOSITORIO,
    "src",
    "models",
    "SARIMAX",
    "resultados_sarimax_walk_forward.csv",
)

FICHERO_PROPHET = os.path.join(
    RAIZ_REPOSITORIO,
    "src",
    "models",
    "Prophet",
    "resultados_prophet_walk_forward_ventana365.csv",
)

CARPETA_REPORTS = os.path.join(RAIZ_REPOSITORIO, "reports")
os.makedirs(CARPETA_REPORTS, exist_ok=True)

FICHERO_COMPARACION_CORTES = os.path.join(
    CARPETA_REPORTS,
    "model_comparison_walk_forward_by_cut.csv",
)

FICHERO_COMPARACION_RESUMEN = os.path.join(
    CARPETA_REPORTS,
    "model_comparison_walk_forward_summary.csv",
)


# =============================================================================
# 2. Funciones auxiliares
# =============================================================================

def cargar_resultados(ruta: str, nombre: str) -> pd.DataFrame:
    """
    Carga un CSV de resultados walk-forward y valida columnas mínimas.
    """
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No se encontró el archivo de resultados de {nombre}: {ruta}"
        )

    df = pd.read_csv(ruta)

    columnas_necesarias = ["corte", "modelo", "mape"]
    faltantes = [col for col in columnas_necesarias if col not in df.columns]

    if faltantes:
        raise ValueError(
            f"El archivo de {nombre} no contiene las columnas necesarias: {faltantes}"
        )

    df = df.copy()
    df["corte"] = pd.to_datetime(df["corte"]).dt.strftime("%Y-%m-%d")
    df["mape"] = pd.to_numeric(df["mape"], errors="coerce")
    df["origen"] = nombre

    return df


def asignar_rol_modelo(modelo: str) -> str:
    """
    Asigna un rol metodológico a cada modelo para facilitar la lectura.
    """
    roles = {
        "Prophet ventana 365": "modelo_candidato",
        "SARIMAX base": "modelo_estadistico_sin_rezagos",
        "SARIMAX + lag 168h": "variante_enriquecida_dependencia_semanal",
        "Baseline lag 24h": "baseline_diagnostico",
        "Baseline lag 168h": "baseline_semanal",
    }

    return roles.get(modelo, "sin_clasificar")


def construir_tabla_por_corte(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye una tabla con cortes en filas y modelos en columnas.
    """
    tabla = df.pivot_table(
        index="corte",
        columns="modelo",
        values="mape",
        aggfunc="mean",
    )

    orden_columnas = [
        "Prophet ventana 365",
        "SARIMAX base",
        "SARIMAX + lag 168h",
        "Baseline lag 24h",
        "Baseline lag 168h",
    ]

    columnas_presentes = [col for col in orden_columnas if col in tabla.columns]
    otras_columnas = [col for col in tabla.columns if col not in columnas_presentes]

    tabla = tabla[columnas_presentes + otras_columnas]
    tabla = tabla.reset_index()

    return tabla.round(2)


def construir_resumen(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye una tabla resumen por modelo.
    """
    resumen = (
        df.groupby("modelo", dropna=False)
        .agg(
            mape_medio=("mape", "mean"),
            mape_std=("mape", "std"),
            mape_min=("mape", "min"),
            mape_max=("mape", "max"),
            n_cortes=("mape", "count"),
        )
        .reset_index()
    )

    resumen["rol_modelo"] = resumen["modelo"].apply(asignar_rol_modelo)

    columnas = [
        "modelo",
        "rol_modelo",
        "mape_medio",
        "mape_std",
        "mape_min",
        "mape_max",
        "n_cortes",
    ]

    resumen = resumen[columnas].sort_values("mape_medio")

    columnas_redondeo = ["mape_medio", "mape_std", "mape_min", "mape_max"]
    resumen[columnas_redondeo] = resumen[columnas_redondeo].round(2)

    return resumen


def validar_cortes_comunes(df: pd.DataFrame) -> None:
    """
    Imprime los cortes disponibles por modelo para detectar comparaciones incompletas.
    """
    print("\nCortes disponibles por modelo")
    print("=" * 80)

    for modelo, grupo in df.groupby("modelo"):
        cortes = sorted(grupo["corte"].dropna().unique())
        print(f"{modelo}: {cortes}")


def main() -> None:
    """
    Ejecuta la comparación de modelos walk-forward.
    """
    print("Cargando resultados SARIMAX...")
    df_sarimax = cargar_resultados(FICHERO_SARIMAX, "SARIMAX")

    print("Cargando resultados Prophet...")
    df_prophet = cargar_resultados(FICHERO_PROPHET, "Prophet")

    print("Unificando resultados...")
    df_total = pd.concat([df_sarimax, df_prophet], ignore_index=True)

    validar_cortes_comunes(df_total)

    tabla_cortes = construir_tabla_por_corte(df_total)
    resumen = construir_resumen(df_total)

    tabla_cortes.to_csv(FICHERO_COMPARACION_CORTES, index=False)
    resumen.to_csv(FICHERO_COMPARACION_RESUMEN, index=False)

    print("\nTabla comparativa por corte")
    print("=" * 80)
    print(tabla_cortes.to_string(index=False))

    print("\nResumen por modelo")
    print("=" * 80)
    print(resumen.to_string(index=False))

    print(f"\nTabla por corte guardada en: {FICHERO_COMPARACION_CORTES}")
    print(f"Resumen guardado en: {FICHERO_COMPARACION_RESUMEN}")


if __name__ == "__main__":
    main()
