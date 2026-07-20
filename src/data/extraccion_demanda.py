#!/usr/bin/env python3
"""
fetch_esios_demanda.py — Ingesta de demanda eléctrica peninsular desde e·sios
============================================================================
TFM: Forecasting de Demanda Energética en España
Máster en Data Science e Inteligencia Artificial — EBIS Business Techschool

OBJETIVO
  Descargar, limpiar y exportar la demanda eléctrica peninsular horaria para
  usarla como variable objetivo del modelo.

FUENTE
  · e·sios / Red Eléctrica
  · Endpoint: https://api.esios.ree.es/indicators/{indicator_id}
  · Indicador recomendado: 1293 — Demanda real
  · Alternativa mencionada en documentación previa: 460 — revisar en e·sios si
    se decide usar otra serie de demanda.

DATASETS GENERADOS
  data/raw/demanda_esios_raw.csv / .parquet
      Respuesta normalizada de e·sios, conservando metadatos útiles.

  data/processed/demanda_peninsular_horaria.csv / .parquet
      Dataset limpio, horario, sin duplicados y preparado para merge por datetime.

COLUMNA CLAVE DE JOIN
  datetime

VARIABLE OBJETIVO
  demanda_mw

NOTA DE SEGURIDAD
  No hardcodear el token en notebooks ni scripts. Define:
      export ESIOS_TOKEN="tu_token"
  o crea un archivo .env local que no se suba a Git.
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import requests
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta


# ═══════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════

BASE_URL = "https://api.esios.ree.es"
DEFAULT_INDICATOR_ID = 1293

# Raíz del repositorio (el script vive en src/data/)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class EsiosConfig:
    token: str
    indicator_id: int = DEFAULT_INDICATOR_ID
    start_date: str = "2021-01-01"
    end_date: Optional[str] = None
    chunk_days: int = 30
    sleep_seconds: float = 0.25
    geo_filter: str = "Península"


class EsiosDemandExtractor:
    """Extractor de demanda eléctrica peninsular desde e·sios."""

    def __init__(self, config: EsiosConfig):
        self.config = config
        self.headers = {
            "Accept": "application/json; application/vnd.esios-api-v1+json",
            "Content-Type": "application/json",
            "Host": "api.esios.ree.es",
            "x-api-key": config.token,
        }

    @staticmethod
    def from_env(
        indicator_id: int = DEFAULT_INDICATOR_ID,
        start_date: str = "2021-01-01",
        end_date: Optional[str] = None,
        chunk_days: int = 30,
        geo_filter: str = "Península",
    ) -> "EsiosDemandExtractor":
        token = os.getenv("ESIOS_TOKEN")
        if not token:
            raise ValueError(
                "No se encontró ESIOS_TOKEN. Define la variable de entorno antes de ejecutar."
            )

        cfg = EsiosConfig(
            token=token,
            indicator_id=indicator_id,
            start_date=start_date,
            end_date=end_date,
            chunk_days=chunk_days,
            geo_filter=geo_filter,
        )
        return EsiosDemandExtractor(cfg)

    def _date_chunks(self) -> Iterable[tuple[pd.Timestamp, pd.Timestamp]]:
        start = pd.Timestamp(self.config.start_date).tz_localize(None)
        end = (
            pd.Timestamp(self.config.end_date).tz_localize(None)
            if self.config.end_date
            else pd.Timestamp.today().tz_localize(None)
        )

        current = start
        while current < end:
            chunk_end = min(current + pd.Timedelta(days=self.config.chunk_days), end)
            yield current, chunk_end
            current = chunk_end

    def fetch_chunk(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        url = f"{BASE_URL}/indicators/{self.config.indicator_id}"
        params = {
            "start_date": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "end_date": end.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        response = requests.get(url, headers=self.headers, params=params, timeout=60)
        response.raise_for_status()

        payload: Dict[str, Any] = response.json()
        values = payload.get("indicator", {}).get("values", [])

        if not values:
            return pd.DataFrame()

        df = pd.DataFrame(values)
        df["indicator_id"] = self.config.indicator_id
        return df

    def fetch_raw(self) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []

        for start, end in self._date_chunks():
            log.info("Descargando e·sios: %s → %s", start, end)
            try:
                chunk = self.fetch_chunk(start, end)
                if not chunk.empty:
                    frames.append(chunk)
            except requests.HTTPError as exc:
                log.error("Error HTTP en %s → %s: %s", start, end, exc)
                raise
            time.sleep(self.config.sleep_seconds)

        if not frames:
            return pd.DataFrame()

        raw = pd.concat(frames, ignore_index=True)
        return raw
    
    def clean(self, raw: pd.DataFrame) -> pd.DataFrame:
        if raw.empty:
            return pd.DataFrame()

        df = raw.copy()

        # datetime de e·sios suele venir con offset horario. Se normaliza a naive
        # local para que encaje con el resto del pipeline si esos datasets usan hora local.
        # Luego se convierte a hora local peninsular sin timezone para joins.
        
        df["datetime_utc"] = pd.to_datetime(
            df["datetime"], 
            errors="coerce",
            utc=True)
        df["datetime"] = (
            df["datetime_utc"]
            .dt.tz_convert("Europe/Madrid")
            .dt.tz_localize(None)
        )

        if "geo_name" in df.columns and self.config.geo_filter:
            mask = df["geo_name"].astype(str).str.contains(
                self.config.geo_filter, case=False, na=False
            )
            if mask.any():
                df = df.loc[mask].copy()

        df = df.rename(columns={"value": "demanda_mw"})
        df["demanda_mw"] = pd.to_numeric(df["demanda_mw"], errors="coerce")

        # datetime local sin timezone para joins consistentes con meteo_horario.
        df["datetime"] = df["datetime"].dt.tz_localize(None)

        keep_cols = [
            "datetime",
            "datetime_utc",
            "demanda_mw",
            "geo_id",
            "geo_name",
            "tz_time",
            "indicator_id",
        ]
        keep_cols = [c for c in keep_cols if c in df.columns]
        df = df[keep_cols]

        df = (
            df.dropna(subset=["datetime", "demanda_mw"])
              .sort_values("datetime")
              .drop_duplicates(subset=["datetime"], keep="last")
              .reset_index(drop=True)
        )

        df["fecha"] = df["datetime"].dt.date.astype(str)
        df["año"] = df["datetime"].dt.year
        df["mes"] = df["datetime"].dt.month
        df["dia"] = df["datetime"].dt.day
        df["hora"] = df["datetime"].dt.hour
        df["dia_semana"] = df["datetime"].dt.dayofweek
        df["es_fin_de_semana"] = df["dia_semana"].isin([5, 6]).astype(int)
        df["fuente"] = "e·sios / Red Eléctrica"

        return df

    @staticmethod
    def quality_report(df: pd.DataFrame) -> Dict[str, Any]:
        if df.empty:
            return {"rows": 0, "status": "empty"}

        expected_hours = pd.date_range(
            df["datetime"].min(),
            df["datetime"].max(),
            freq="h",
        )

        missing_hours = expected_hours.difference(pd.DatetimeIndex(df["datetime"]))

        return {
            "rows": len(df),
            "start": str(df["datetime"].min()),
            "end": str(df["datetime"].max()),
            "duplicated_datetime": int(df.duplicated("datetime").sum()),
            "missing_hours": int(len(missing_hours)),
            "null_demanda_mw": int(df["demanda_mw"].isna().sum()),
            "min_demanda_mw": float(df["demanda_mw"].min()),
            "max_demanda_mw": float(df["demanda_mw"].max()),
            "mean_demanda_mw": float(df["demanda_mw"].mean()),
        }

    @staticmethod
    def save(df: pd.DataFrame, folder: Path, name: str) -> None:
        if df.empty:
            log.warning("Dataset vacío, no se guarda: %s", name)
            return

        csv_path = folder / f"{name}.csv"
        parquet_path = folder / f"{name}.parquet"

        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        df.to_parquet(parquet_path, index=False)

        log.info("Guardado: %s (%d filas × %d columnas)", csv_path, len(df), len(df.columns))
        log.info("Guardado: %s", parquet_path)

    def run(self) -> Dict[str, pd.DataFrame]:
        log.info("TFM — Ingesta demanda e·sios")
        log.info("Indicador: %s", self.config.indicator_id)
        log.info("Ventana: %s → %s", self.config.start_date, self.config.end_date or "hoy")

        raw = self.fetch_raw()
        clean = self.clean(raw)
        report = self.quality_report(clean)

        log.info("QA demanda: %s", report)

        self.save(raw, RAW_DIR, "demanda_esios_raw")
        # A data/raw: es el punto de entrada de feature_engineering.py
        self.save(clean, RAW_DIR, "demanda_peninsular_horaria")

        return {
            "demanda_esios_raw": raw,
            "demanda_peninsular_horaria": clean,
        }


if __name__ == "__main__":
    extractor = EsiosDemandExtractor.from_env(
        indicator_id=1293,
        start_date="2021-01-01",
        end_date=None,
        chunk_days=30,
        geo_filter="Península",
    )
    extractor.run()
