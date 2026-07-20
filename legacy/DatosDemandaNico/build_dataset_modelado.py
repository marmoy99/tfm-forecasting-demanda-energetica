#!/usr/bin/env python3
"""
build_dataset_modelado.py — Ensamble inicial para modelado del TFM

Une:
  1. demanda_peninsular_horaria
  2. temperatura_nacional_ponderada
  3. opcionalmente meteo_horario agregado

Output:
  data/processed/dataset_modelado_base.parquet
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
    df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)
    df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)
    return df


def add_lags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("datetime").copy()
    df["demanda_lag_24h"] = df["demanda_mw"].shift(24)
    df["demanda_lag_168h"] = df["demanda_mw"].shift(168)
    df["demanda_rolling_24h"] = df["demanda_mw"].shift(1).rolling(24).mean()
    df["demanda_rolling_168h"] = df["demanda_mw"].shift(1).rolling(168).mean()
    return df


def build_dataset(
    demanda_path: Path = PROCESSED_DIR / "demanda_peninsular_horaria.parquet",
    temperatura_path: Path = PROCESSED_DIR / "temperatura_nacional_ponderada.parquet",
    output_path: Path = PROCESSED_DIR / "dataset_modelado_base.parquet",
) -> pd.DataFrame:
    demanda = pd.read_parquet(demanda_path)
    temp = pd.read_parquet(temperatura_path)

    demanda["datetime"] = pd.to_datetime(demanda["datetime"])
    demanda["fecha"] = pd.to_datetime(demanda["fecha"]).dt.date.astype(str)
    temp["fecha"] = pd.to_datetime(temp["fecha"]).dt.date.astype(str)

    df = demanda.merge(temp, on="fecha", how="left", validate="many_to_one")

    df = add_cyclical_time_features(df)
    df = add_lags(df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    df.to_csv(output_path.with_suffix(".csv"), index=False, encoding="utf-8-sig")

    return df


if __name__ == "__main__":
    dataset = build_dataset()
    print(dataset.shape)
    print(dataset.head())
