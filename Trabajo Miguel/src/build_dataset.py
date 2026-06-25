#!/usr/bin/env python3
"""
build_dataset.py — Dataset de modelado limpio para el TFM
==========================================================
TFM: Forecasting de Demanda Energética en España

Produce un dataset horario con 16 columnas prácticas y significativas.

Columnas finales:
  datetime, demanda_mw,
  hora, dia_semana, mes, es_fin_de_semana, es_festivo, es_agosto,
  t_nac, HDD, CDD,
  radiacion_solar, humedad_relativa, velocidad_viento,
  demanda_lag_24h, demanda_lag_168h

Output:
  data/processed/dataset_modelado.parquet
  data/processed/dataset_modelado.csv
"""

import pandas as pd
import numpy as np
import holidays
from pathlib import Path
from datetime import date, timedelta
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAW_DIR       = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

DEMAND_CSV       = RAW_DIR  / "demanda_peninsular_horaria.csv"
METEO_PQRT       = PROCESSED_DIR / "temperatura_nacional_ponderada.parquet"
METEO_HORARIO    = RAW_DIR  / "meteo_horario.parquet"
OUTPUT_NAME      = "dataset_modelado"

COMFORT_TEMP = 18.0


# ═══════════════════════════════════════════════════════════════════
# CARGA
# ═══════════════════════════════════════════════════════════════════

def load_demand() -> pd.DataFrame:
    log.info("Cargando demanda REE…")
    df = pd.read_csv(DEMAND_CSV, parse_dates=["datetime"])
    df = df[["datetime", "demanda_mw", "hora", "dia_semana", "mes", "es_fin_de_semana"]].copy()
    df = df.sort_values("datetime").reset_index(drop=True)
    log.info("  %d filas  |  %s → %s", len(df), df["datetime"].min(), df["datetime"].max())
    return df


def load_meteo_extra() -> pd.DataFrame:
    """
    Carga radiacion_solar, humedad_relativa y velocidad_viento de meteo_horario.
    Agrega las 7 ciudades con media simple por datetime — para estas variables
    la media geográfica es más representativa que la ponderación por población
    (la radiación y el viento dependen del territorio, no de dónde vive la gente).
    """
    if not METEO_HORARIO.exists():
        log.warning("meteo_horario.parquet no encontrado — variables extra omitidas.")
        return pd.DataFrame()

    log.info("Cargando radiación, humedad y viento (meteo_horario)…")
    df = pd.read_parquet(METEO_HORARIO, columns=["datetime", "radiacion_solar",
                                                   "humedad_relativa", "velocidad_viento"])
    df["datetime"] = pd.to_datetime(df["datetime"])

    df = (df.groupby("datetime", as_index=False)
            .agg(radiacion_solar  = ("radiacion_solar",  "mean"),
                 humedad_relativa = ("humedad_relativa",  "mean"),
                 velocidad_viento = ("velocidad_viento",  "mean"))
            .round(2))

    log.info("  %d filas horarias agregadas.", len(df))
    return df


def load_meteo() -> pd.DataFrame:
    if not METEO_PQRT.exists():
        log.warning("temperatura_nacional_ponderada.parquet no encontrado.")
        return pd.DataFrame()

    log.info("Cargando temperatura nacional ponderada…")
    df = pd.read_parquet(METEO_PQRT)

    # Compatibilidad con formato diario (versión antigua del extractor)
    if "datetime" not in df.columns:
        log.info("  Formato diario — expandiendo a granularidad horaria…")
        df = df.rename(columns={"t_nac_media": "t_nac"})
        df["fecha"] = pd.to_datetime(df["fecha"])
        horas = pd.DataFrame({"_h": range(24)})
        df = df.merge(horas, how="cross")
        df["datetime"] = df["fecha"] + pd.to_timedelta(df["_h"], unit="h")
        df = df.drop(columns=["_h"])

    df["datetime"] = pd.to_datetime(df["datetime"])
    keep = [c for c in ["datetime", "t_nac", "HDD", "CDD"] if c in df.columns]
    return df[keep].copy()


# ═══════════════════════════════════════════════════════════════════
# CALENDARIO
# ═══════════════════════════════════════════════════════════════════

def build_calendar(datetimes: pd.Series) -> pd.DataFrame:
    """
    Genera dos variables de calendario:

    es_dia_especial  →  agrupa festivos, puentes, Semana Santa y Navidad
                        en un único indicador. Todos comparten el mismo
                        efecto sobre la demanda: caída respecto al patrón
                        normal de ese día de la semana. No tiene sentido
                        tener 4 columnas separadas para el modelo.

    es_agosto        →  se mantiene separado porque su efecto es distinto:
                        no es un día puntual sino un mes entero de caída
                        estructural por vacaciones industriales (~15% menos).
    """
    log.info("Construyendo calendario…")
    dates  = pd.to_datetime(datetimes)
    years  = dates.dt.year.unique().tolist()
    es_hol = holidays.Spain(years=years)

    df = pd.DataFrame({"datetime": dates})
    df["_date"] = df["datetime"].dt.date

    # Festivo nacional
    holiday_set = set(es_hol.keys())
    df["_festivo"] = df["_date"].map(lambda d: d in holiday_set).astype(int)

    # Semana Santa (Domingo de Ramos → Domingo de Resurrección)
    def in_semana_santa(d):
        easter = _easter(d.year)
        return int(easter - timedelta(days=7) <= d <= easter)
    df["_ss"] = df["_date"].map(in_semana_santa)

    # Navidad (24 dic – 6 ene)
    df["_nav"] = df["_date"].map(
        lambda d: int((d.month == 12 and d.day >= 24) or (d.month == 1 and d.day <= 6))
    )

    # Puente
    def is_puente(d):
        if d.weekday() >= 5 or d in holiday_set:
            return 0
        return int(
            (d - timedelta(1)).weekday() >= 5 or (d - timedelta(1)) in holiday_set or
            (d + timedelta(1)).weekday() >= 5 or (d + timedelta(1)) in holiday_set
        )
    df["_puente"] = df["_date"].map(is_puente)

    df["es_festivo"] = df["_festivo"]
    df["es_agosto"]  = (df["datetime"].dt.month == 8).astype(int)

    return df[["datetime", "es_festivo", "es_agosto"]].copy()


def _easter(year: int) -> date:
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(114 + h + l - 7 * m, 31)
    return date(year, month, day + 1)


# ═══════════════════════════════════════════════════════════════════
# LAGS
# ═══════════════════════════════════════════════════════════════════

def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Solo los dos lags más predictivos:
      lag_24h  → correlación 0.81  (misma hora ayer)
      lag_168h → correlación 0.88  (misma hora semana pasada) — el más potente

    Se descarta lag_48h (0.63, redundante con los anteriores) y las medias
    móviles (redundantes con los lags + añaden complejidad sin ganancia clara).
    """
    log.info("Calculando lags de demanda…")
    df = df.sort_values("datetime").reset_index(drop=True)
    df["demanda_lag_24h"]  = df["demanda_mw"].shift(24)
    df["demanda_lag_168h"] = df["demanda_mw"].shift(168)
    return df


# ═══════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════

def build() -> pd.DataFrame:
    sep = "═" * 62
    log.info(sep)
    log.info("  TFM — build_dataset.py")
    log.info(sep)

    demand_df    = load_demand()
    meteo_df     = load_meteo()
    meteo_ext_df = load_meteo_extra()
    cal_df       = build_calendar(demand_df["datetime"])

    # JOINs
    df = demand_df.merge(meteo_df,     on="datetime", how="left") if not meteo_df.empty     else demand_df.copy()
    df = df.merge(meteo_ext_df,        on="datetime", how="left") if not meteo_ext_df.empty else df
    df = df.merge(cal_df,              on="datetime", how="left")

    # Lags de demanda
    df = add_lags(df)

    # Orden final de columnas — legible y lógico
    columnas_finales = [
        "datetime",           # fecha y hora
        "demanda_mw",         # consumo eléctrico peninsular (MW)
        "hora",               # hora del día (0-23)
        "dia_semana",         # 0=lunes … 6=domingo
        "mes",                # mes (1-12)
        "es_fin_de_semana",   # 1 si sábado o domingo
        "es_festivo",         # 1 si festivo nacional
        "es_agosto",          # 1 si agosto (vacaciones industriales)
        "t_nac",              # temperatura nacional ponderada por población (°C)
        "HDD",                # grados-día calefacción: max(0, 18 - t_nac)
        "CDD",                # grados-día refrigeración: max(0, t_nac - 18)
        "radiacion_solar",    # W/m² media nacional — proxy de generación fotovoltaica
        "humedad_relativa",   # % media nacional — amplifica efecto del calor en AC
        "velocidad_viento",   # km/h media nacional — proxy de generación eólica
        "demanda_lag_24h",    # demanda misma hora ayer (corr 0.81)
        "demanda_lag_168h",   # demanda misma hora semana pasada (corr 0.88)
    ]
    df = df[[c for c in columnas_finales if c in df.columns]]

    # Reporte
    log.info("\n%s", sep)
    log.info("  DATASET FINAL")
    log.info(sep)
    log.info("  Filas    : %d", len(df))
    log.info("  Columnas : %d  →  %s", len(df.columns), ", ".join(df.columns.tolist()))
    log.info("  Período  : %s → %s", df["datetime"].min(), df["datetime"].max())
    log.info("  Nulos    : %d", df.isna().sum().sum())

    # Correlaciones finales
    corr = df.drop(columns=["datetime"]).corrwith(df["demanda_mw"]).abs().sort_values(ascending=False)
    log.info("\n  Correlación con demanda_mw:")
    for col, val in corr.items():
        if col != "demanda_mw":
            bar = "█" * int(val * 30)
            log.info("  %-22s %.3f  %s", col, val, bar)

    # Exportar
    out_p = PROCESSED_DIR / f"{OUTPUT_NAME}.parquet"
    out_c = PROCESSED_DIR / f"{OUTPUT_NAME}.csv"
    df.to_parquet(out_p, index=False)
    df.to_csv(out_c, index=False, encoding="utf-8-sig")
    log.info("\n  ✓ %s", out_p)
    log.info("  ✓ %s", out_c)
    log.info(sep)

    return df


if __name__ == "__main__":
    build()
