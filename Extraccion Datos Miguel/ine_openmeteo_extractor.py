#!/usr/bin/env python3
"""
fetch_meteo_ine.py — Ingesta meteorológica + demográfica para el TFM
======================================================================
TFM: Forecasting de Demanda Energética en España
Máster en Data Science e Inteligencia Artificial — EBIS Business Techschool

FUENTES
  · INE  API TEMPUS   https://servicios.ine.es/wstempus/js/ES/
  · Open-Meteo Archive https://archive-api.open-meteo.com/v1/archive

DATASETS GENERADOS  (data/raw/ y data/processed/)
  meteo_horario.csv / .parquet         Variables horarias por ciudad
  meteo_diario.csv  / .parquet         Resumen diario + amanecer/atardecer
  poblacion_ciudades.csv               Población anual INE por ciudad/provincia
  temperatura_nacional_ponderada.csv   Feature clave: T ponderada por población

DECISIONES DE DISEÑO
  · Ciudades: 7 capitales peninsulares por masa de población (cubren ~40% de la
    población española y los principales patrones de consumo: clima continental,
    mediterráneo, atlántico y semiárido).
  · Las Palmas y Palma excluidas: mercados eléctricos no peninsulares (sistemas
    insulares independientes de REE-peninsular).
  · Ventana temporal: últimos 5 años completos desde hoy. Se evita 2020 (COVID),
    año con una caída ~15% de demanda que distorsionaría el modelo.
  · HDD/CDD (grados-día de calefacción/refrigeración): derivados directamente
    de la temperatura media diaria ponderada. Son mejores features que la
    temperatura bruta porque capturan la relación no lineal con el consumo HVAC.
  · Temperatura ponderada por población (T_nac): media nacional "sintética"
    que agrega las 7 ciudades con peso proporcional a su población. Es la
    variable meteorológica más correlacionada con la demanda eléctrica total.
  · apparent_temperature (sensación térmica): incluida en datos horarios porque
    predice mejor el uso de calefacción/AC que la temperatura real.
  · shortwave_radiation (radiación solar): relevante tanto para demanda
    (refrigeración) como para estimación de generación fotovoltaica (demanda neta).

ENCAJE CON EL PIPELINE DEL TFM
  Este script genera las variables meteorológicas que build_dataset.py fusionará
  con la demanda horaria de REE (e·sios indicador 460) indexada por datetime.
  Columna clave de join: datetime (UTC+1/CEST según período).

RECOMENDACIONES ADICIONALES
  1. Complementar con AEMET OpenData para validación cruzada de temperatura.
     Open-Meteo usa ERA5 (reanálisis ECMWF) — ligeramente más suavizado que
     observaciones de estación pero sin huecos y con cobertura histórica total.
  2. Para el modelo LightGBM añadir lags de temperatura: T_t-24h y T_t-168h
     (misma hora ayer/semana pasada), que capturan inercia térmica.
  3. Considerar wind_power_density (∝ viento³) además de wind_speed para
     correlacionar con generación eólica y demanda neta.
  4. El período 2021-presente incluye la crisis energética 2021-2022 (precios
     disparados), lo que puede introducir cambios estructurales en elasticidad
     precio-demanda. Considerar variable binaria is_crisis_energetica.
"""

import requests
import pandas as pd
import numpy as np
from datetime import date, timedelta
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════

# 7 capitales peninsulares con mayor masa de población.
# Cubren clima continental (Madrid, Zaragoza), mediterráneo (Barcelona,
# Valencia, Málaga), atlántico (Bilbao) y meridional seco (Sevilla).
# Las coordenadas apuntan a estaciones meteorológicas urbanas representativas.
CITIES: Dict[str, Dict] = {
    "Madrid": {
        "lat": 40.4168, "lon": -3.7038,
        "cod_prov": "28",
        "nombre_prov": "Madrid",
    },
    "Barcelona": {
        "lat": 41.3851, "lon": 2.1734,
        "cod_prov": "08",
        "nombre_prov": "Barcelona",
    },
    "Valencia": {
        "lat": 39.4699, "lon": -0.3763,
        "cod_prov": "46",
        "nombre_prov": "Valencia/València",
    },
    "Sevilla": {
        "lat": 37.3891, "lon": -5.9845,
        "cod_prov": "41",
        "nombre_prov": "Sevilla",
    },
    "Zaragoza": {
        "lat": 41.6488, "lon": -0.8891,
        "cod_prov": "50",
        "nombre_prov": "Zaragoza",
    },
    "Málaga": {
        "lat": 36.7213, "lon": -4.4214,
        "cod_prov": "29",
        "nombre_prov": "Málaga",
    },
    "Bilbao": {
        "lat": 43.2630, "lon": -2.9350,
        "cod_prov": "48",
        "nombre_prov": "Bizkaia",
    },
}

# Ventana temporal: 5 años completos hacia atrás desde hoy.
# Arranca el 1 de enero del año (today.year - 5) para tener años completos.
# Para today = 2026-06-02 → START = 2021-01-01.
# Esto deja fuera 2020 (COVID: -15% demanda) y los datos más antiguos
# que harían el modelo menos representativo del comportamiento actual.
_today = date.today()
START_DATE: str = date(_today.year - 5, 1, 1).isoformat()
END_DATE: str   = (_today - timedelta(days=5)).isoformat()   # límite API archivo

# Temperatura de confort para HDD/CDD (estándar energético europeo)
COMFORT_TEMP: float = 18.0

# Outputs
OUTPUT_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# IDs tablas INE — Padrón Municipal Continuo (operación 22)
INE_TABLE_POBLACION_PROV      = 2852   # Población por provincias y sexo
INE_TABLE_VARIACION_CAPITALES = 2919   # Variación anual — capitales


# ═══════════════════════════════════════════════════════════════════
# CLIENTE INE
# ═══════════════════════════════════════════════════════════════════

class INEClient:
    """
    API TEMPUS del INE — datos anuales del Padrón Municipal.
    Se usa únicamente para obtener la población de cada ciudad y calcular
    los pesos de la temperatura nacional ponderada (feature 5.4 del TFM).
    """

    BASE_URL = "https://servicios.ine.es/wstempus/js/ES"
    DELAY    = 0.6

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "TFM-EnergyForecast/1.0",
        })

    def _get(self, endpoint: str, params: dict | None = None) -> Any:
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            resp = self.session.get(url, params=params, timeout=40)
            resp.raise_for_status()
            time.sleep(self.DELAY)
            return resp.json()
        except requests.RequestException as exc:
            log.error("INE error [%s]: %s", endpoint, exc)
            return None

    @staticmethod
    def _parse(raw: List[Dict]) -> pd.DataFrame:
        records: List[Dict] = []
        for serie in raw:
            nombre = serie.get("Nombre", "")
            for p in serie.get("Data", []):
                if p.get("Secreto", False) or p.get("Valor") is None:
                    continue
                records.append({
                    "serie":   nombre,
                    "año":     p["Anyo"],
                    "periodo": p.get("FK_Periodo"),
                    "valor":   p["Valor"],
                })
        return pd.DataFrame(records)

    def get_poblacion_provincial(self) -> pd.DataFrame:
        """
        Tabla 2852 — Población total por provincia y sexo (Padrón Continuo).
        Se filtra a 'Ambos sexos' y a las provincias de las ciudades del estudio.
        """
        log.info("INE ▶ Población por provincia (tabla 2852)…")
        raw = self._get(f"DATOS_TABLA/{INE_TABLE_POBLACION_PROV}", {"nult": 300})
        if not raw:
            return pd.DataFrame()

        df = self._parse(raw)

        # El nombre de serie tiene formato "Provincia. Ambos sexos" / "Hombres" / "Mujeres"
        partes      = df["serie"].str.split(r"\.\s*", n=1, expand=True)
        df["provincia_raw"] = partes[0].str.strip()
        df["sexo"]          = partes[1].str.strip() if partes.shape[1] > 1 else "Ambos sexos"

        # Solo total poblacional
        df = df[df["sexo"].str.contains("Ambos|Total", case=False, na=False)].copy()

        # Mapeo provincia_raw → ciudad del estudio
        prov_to_city = {cfg["nombre_prov"]: city for city, cfg in CITIES.items()}
        df["ciudad"] = df["provincia_raw"].map(prov_to_city)
        df = df[df["ciudad"].notna()].copy()

        # Solo años relevantes
        start_yr = int(START_DATE[:4])
        df = df[df["año"].between(start_yr, _today.year)].copy()

        df = df.rename(columns={"valor": "poblacion"})
        df["fuente"] = "INE – Padrón Municipal Continuo"

        return df[["año", "ciudad", "provincia_raw", "poblacion", "fuente"]].copy()

    def get_variacion_poblacion(self) -> pd.DataFrame:
        """
        Tabla 2919 — Variación anual en capitales (nacimientos, defunciones,
        saldo migratorio). Contextualiza cambios estructurales de demanda.
        """
        log.info("INE ▶ Variación anual de población — capitales (tabla 2919)…")
        raw = self._get(f"DATOS_TABLA/{INE_TABLE_VARIACION_CAPITALES}", {"nult": 200})
        if not raw:
            return pd.DataFrame()

        df = self._parse(raw)
        partes         = df["serie"].str.split(r"\.\s*", n=1, expand=True)
        df["municipio"] = partes[0].str.strip()
        df["indicador"] = partes[1].str.strip() if partes.shape[1] > 1 else "Total"

        start_yr = int(START_DATE[:4])
        df = df[df["año"].between(start_yr, _today.year)].copy()

        city_names = set(CITIES.keys())
        df = df[df["municipio"].isin(city_names)].copy()

        df = df.rename(columns={"valor": "variacion"})
        df["fuente"] = "INE – Padrón Municipal Continuo"

        return df[["año", "municipio", "indicador", "variacion", "fuente"]].copy()


# ═══════════════════════════════════════════════════════════════════
# CLIENTE OPEN-METEO
# ═══════════════════════════════════════════════════════════════════

class OpenMeteoClient:
    """
    API de archivo histórico de Open-Meteo (ERA5/ERA5-Land).
    Ventajas frente a AEMET para este TFM:
      · Sin límite de llamadas (AEMET tiene rate limits estrictos)
      · Sin huecos temporales (reanálisis completo)
      · Variables de radiación solar y sensación térmica directamente disponibles
    """

    ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
    DELAY       = 0.4

    # Variables horarias: temperatura, humedad, viento, radiación, sensación
    HOURLY_VARS = [
        "temperature_2m",          # temperatura a 2 m (°C)
        "apparent_temperature",    # sensación térmica — mejor proxy HVAC que T real
        "relative_humidity_2m",    # humedad relativa (%)
        "wind_speed_10m",          # velocidad viento (km/h)
        "shortwave_radiation",     # radiación solar global (W/m²) — PV + refrigeración
        "precipitation",           # precipitación (mm)
    ]

    # Variables diarias: resumen + amanecer/atardecer
    DAILY_VARS = [
        "temperature_2m_max",
        "temperature_2m_min",
        "temperature_2m_mean",
        "apparent_temperature_max",
        "apparent_temperature_min",
        "sunrise",
        "sunset",
        "shortwave_radiation_sum",  # radiación acumulada diaria (MJ/m²)
        "wind_speed_10m_max",
        "precipitation_sum",
    ]

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "TFM-EnergyForecast/1.0"})

    def _fetch(self, lat: float, lon: float, variables: List[str],
               granularity: str) -> dict | None:
        params = {
            "latitude":   lat,
            "longitude":  lon,
            "start_date": START_DATE,
            "end_date":   END_DATE,
            granularity:  ",".join(variables),
            "timezone":   "Europe/Madrid",
        }
        try:
            resp = self.session.get(self.ARCHIVE_URL, params=params, timeout=120)
            resp.raise_for_status()
            time.sleep(self.DELAY)
            return resp.json()
        except requests.RequestException as exc:
            log.error("Open-Meteo error: %s", exc)
            return None

    # ── Datos horarios ─────────────────────────────────────────────
    def get_hourly(self, city: str, lat: float, lon: float) -> pd.DataFrame:
        log.info("Open-Meteo ▶ Datos horarios para %s (%s → %s)…", city, START_DATE, END_DATE)
        data = self._fetch(lat, lon, self.HOURLY_VARS, "hourly")
        if not data or "hourly" not in data:
            log.warning("Sin datos horarios para %s.", city)
            return pd.DataFrame()

        h  = data["hourly"]
        dt = pd.to_datetime(h["time"])

        df = pd.DataFrame({
            "datetime":            dt,
            "fecha":               dt.date,
            "hora":                dt.strftime("%H:%M"),
            "ciudad":              city,
            "latitud":             data.get("latitude",  lat),
            "longitud":            data.get("longitude", lon),
            "temperatura":         h.get("temperature_2m"),
            "sensacion_termica":   h.get("apparent_temperature"),
            "humedad_relativa":    h.get("relative_humidity_2m"),
            "velocidad_viento":    h.get("wind_speed_10m"),
            "radiacion_solar":     h.get("shortwave_radiation"),
            "precipitacion":       h.get("precipitation"),
        })

        df["año"]       = dt.year
        df["mes"]       = dt.month
        df["hora_num"]  = dt.hour
        df["fuente"]    = "Open-Meteo (ERA5)"

        return df

    # ── Datos diarios ──────────────────────────────────────────────
    def get_daily(self, city: str, lat: float, lon: float) -> pd.DataFrame:
        log.info("Open-Meteo ▶ Datos diarios para %s…", city)
        data = self._fetch(lat, lon, self.DAILY_VARS, "daily")
        if not data or "daily" not in data:
            log.warning("Sin datos diarios para %s.", city)
            return pd.DataFrame()

        d  = data["daily"]
        dt = pd.to_datetime(d["time"])

        df = pd.DataFrame({
            "fecha":                  dt,
            "ciudad":                 city,
            "latitud":                data.get("latitude",  lat),
            "longitud":               data.get("longitude", lon),
            "temperatura_max":        d.get("temperature_2m_max"),
            "temperatura_min":        d.get("temperature_2m_min"),
            "temperatura_media":      d.get("temperature_2m_mean"),
            "sensacion_max":          d.get("apparent_temperature_max"),
            "sensacion_min":          d.get("apparent_temperature_min"),
            "hora_amanecer":          d.get("sunrise"),
            "hora_atardecer":         d.get("sunset"),
            "radiacion_acumulada":    d.get("shortwave_radiation_sum"),
            "viento_max":             d.get("wind_speed_10m_max"),
            "precipitacion_total":    d.get("precipitation_sum"),
        })

        # Hora legible HH:MM
        for col in ("hora_amanecer", "hora_atardecer"):
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%H:%M")

        df["año"]       = dt.year
        df["mes"]       = dt.month
        df["dia"]       = dt.day
        df["fuente"]    = "Open-Meteo (ERA5)"

        return df


# ═══════════════════════════════════════════════════════════════════
# PROCESADOR DE FEATURES
# ═══════════════════════════════════════════════════════════════════

class FeatureBuilder:
    """
    Construye las features meteorológicas listas para el pipeline del TFM.
    Encaja con el rol de build_dataset.py: recibe los datos crudos y genera
    variables derivadas ya documentadas en la sección 5.4 del plan.
    """

    # ── Temperatura nacional ponderada por población ───────────────
    @staticmethod
    def compute_national_temperature(
        daily_df: pd.DataFrame,
        pop_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Feature 5.4 TFM: "Temperatura media ponderada por población
        de las principales ciudades."

        Para cada día calcula T_nac = Σ(T_ciudad_i × peso_i)
        donde peso_i = población_i / Σ población.

        También calcula la sensación térmica ponderada y los HDD/CDD.
        """
        if daily_df.empty or pop_df.empty:
            log.warning("Datos insuficientes para temperatura nacional ponderada.")
            return pd.DataFrame()

        # Último año de población disponible por ciudad
        pop_latest = (
            pop_df
            .sort_values("año")
            .groupby("ciudad")["poblacion"]
            .last()
            .reset_index()
        )
        total_pop = pop_latest["poblacion"].sum()
        pop_latest["peso"] = pop_latest["poblacion"] / total_pop

        log.info("Pesos de temperatura nacional ponderada:")
        for _, row in pop_latest.iterrows():
            log.info("  %-12s  %6.1fk hab  peso = %.4f",
                     row["ciudad"], row["poblacion"] / 1000, row["peso"])

        # Unir pesos a datos diarios
        merged = daily_df.merge(pop_latest[["ciudad", "peso"]], on="ciudad", how="left")
        merged["temp_pond"]    = merged["temperatura_media"] * merged["peso"]
        merged["sensacion_pond"] = merged["sensacion_max"].fillna(merged["temperatura_max"]) * merged["peso"]

        # Agregar por fecha → temperatura nacional
        national = (
            merged
            .groupby("fecha", as_index=False)
            .agg(
                t_nac_media    = ("temp_pond",     "sum"),
                t_nac_max      = ("temperatura_max",   lambda x: (x * merged.loc[x.index, "peso"]).sum()),
                t_nac_min      = ("temperatura_min",   lambda x: (x * merged.loc[x.index, "peso"]).sum()),
                sensacion_nac  = ("sensacion_pond", "sum"),
                ciudades_usadas= ("ciudad",         "nunique"),
            )
        )

        # HDD y CDD — grados-día de calefacción y refrigeración (base 18 °C)
        national["HDD"] = np.maximum(0.0, COMFORT_TEMP - national["t_nac_media"])
        national["CDD"] = np.maximum(0.0, national["t_nac_media"] - COMFORT_TEMP)

        # Acumulados mensuales (útiles para análisis estacional)
        national["fecha"]  = pd.to_datetime(national["fecha"])
        national["año"]    = national["fecha"].dt.year
        national["mes"]    = national["fecha"].dt.month

        national["fuente"] = "Open-Meteo (ERA5) + INE Padrón"

        return national

    # ── HDD/CDD por ciudad ─────────────────────────────────────────
    @staticmethod
    def add_hdd_cdd(daily_df: pd.DataFrame) -> pd.DataFrame:
        """Añade HDD y CDD a datos diarios por ciudad."""
        df = daily_df.copy()
        df["HDD"] = np.maximum(0.0, COMFORT_TEMP - df["temperatura_media"].fillna(df["temperatura_max"]))
        df["CDD"] = np.maximum(0.0, df["temperatura_media"].fillna(df["temperatura_max"]) - COMFORT_TEMP)
        return df


# ═══════════════════════════════════════════════════════════════════
# ORQUESTADOR
# ═══════════════════════════════════════════════════════════════════

class Extractor:
    """Orquesta la extracción, procesado y exportación."""

    def __init__(self) -> None:
        self.ine    = INEClient()
        self.meteo  = OpenMeteoClient()
        self.feats  = FeatureBuilder()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Recolección ────────────────────────────────────────────────
    def collect_weather(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        hourly_frames, daily_frames = [], []
        for city, cfg in CITIES.items():
            h = self.meteo.get_hourly(city, cfg["lat"], cfg["lon"])
            d = self.meteo.get_daily(city,  cfg["lat"], cfg["lon"])
            if not h.empty:
                hourly_frames.append(h)
            if not d.empty:
                daily_frames.append(d)

        hourly = pd.concat(hourly_frames, ignore_index=True) if hourly_frames else pd.DataFrame()
        daily  = pd.concat(daily_frames,  ignore_index=True) if daily_frames  else pd.DataFrame()
        return hourly, daily

    def collect_population(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        pop = self.ine.get_poblacion_provincial()
        var = self.ine.get_variacion_poblacion()
        return pop, var

    # ── Exportación ────────────────────────────────────────────────
    @staticmethod
    def _save(df: pd.DataFrame, folder: Path, name: str) -> None:
        if df.empty:
            log.warning("Dataset vacío, no se guarda: %s", name)
            return
        csv_path     = folder / f"{name}.csv"
        parquet_path = folder / f"{name}.parquet"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        df.to_parquet(parquet_path, index=False)
        log.info("  ✓  %-52s  %7d filas × %d cols",
                 str(parquet_path), len(df), len(df.columns))

    # ── Pipeline completo ──────────────────────────────────────────
    def run(self) -> Dict[str, pd.DataFrame]:
        sep = "═" * 66
        log.info(sep)
        log.info("  TFM — Ingesta Meteorológica + Demográfica")
        log.info("  Ventana : %s → %s", START_DATE, END_DATE)
        log.info("  Ciudades: %s", ", ".join(CITIES))
        log.info(sep)

        # 1 ── Clima ────────────────────────────────────────────────
        log.info("\n[1/4] Datos meteorológicos (Open-Meteo ERA5)…")
        hourly_df, daily_df = self.collect_weather()
        if not daily_df.empty:
            daily_df = self.feats.add_hdd_cdd(daily_df)

        # 2 ── Población INE ────────────────────────────────────────
        log.info("\n[2/4] Datos demográficos (INE Padrón)…")
        pop_df, var_df = self.collect_population()

        # 3 ── Temperatura nacional ponderada (feature principal) ───
        log.info("\n[3/4] Temperatura nacional ponderada por población…")
        t_nac_df = self.feats.compute_national_temperature(daily_df, pop_df)

        # 4 ── Guardar ──────────────────────────────────────────────
        log.info("\n[4/4] Exportando datasets…")
        outputs = {
            "meteo_horario":                (OUTPUT_DIR,    hourly_df),
            "meteo_diario":                 (OUTPUT_DIR,    daily_df),
            "poblacion_ciudades":           (OUTPUT_DIR,    pop_df),
            "variacion_poblacion":          (OUTPUT_DIR,    var_df),
            "temperatura_nacional_ponderada": (PROCESSED_DIR, t_nac_df),
        }
        results: Dict[str, pd.DataFrame] = {}
        for name, (folder, df) in outputs.items():
            self._save(df, folder, name)
            results[name] = df

        # ── Resumen ────────────────────────────────────────────────
        log.info("\n" + sep)
        log.info("  EXTRACCIÓN COMPLETADA")
        log.info(sep)
        for name, df in results.items():
            if not df.empty:
                log.info("  %-42s  %7d filas", name, len(df))
            else:
                log.info("  %-42s  (vacío)", name)

        log.info("\n  PRÓXIMOS PASOS:")
        log.info("  · fetch_ree.py      → demanda horaria REE (indicador 460)")
        log.info("  · build_dataset.py  → merge por datetime con meteo_horario.parquet")
        log.info("  · feature_eng.py    → añadir lags T_t-24h, T_t-168h; festivos")
        log.info(sep)

        return results


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    extractor = Extractor()
    datasets  = extractor.run()
