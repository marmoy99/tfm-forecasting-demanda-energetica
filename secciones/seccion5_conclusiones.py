#!/usr/bin/env python3
"""
Sección 5 — Conclusiones
========================
Cierre del TFM: resultado frente al objetivo, modelo seleccionado,
recorrido metodológico, limitaciones y líneas futuras.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from theme import section_title, kpi_row

REPORTS = Path(__file__).resolve().parents[1] / "reports"
OBJETIVO_MAPE = 3.0
SARIMAX_VARIANTE = "SARIMAX armónico + lag 168h"


@st.cache_data(show_spinner=False)
def resumen_mape() -> dict:
    lgb = pd.read_csv(REPORTS / "model_results" / "resultados_lightgbm_walk_forward.csv")["mape"].mean()
    pro = pd.read_csv(REPORTS / "model_results" / "resultados_prophet_walk_forward.csv")["mape"].mean()
    sar = pd.read_csv(REPORTS / "model_comparison" / "resultados_sarimax_comparacion.csv",
                      encoding="utf-8-sig")
    sar = sar[sar["modelo"] == SARIMAX_VARIANTE]["mape"].astype(float).mean()
    return {"LightGBM": float(lgb), "Prophet": float(pro), "SARIMAX": float(sar)}


def render(df_dataset: pd.DataFrame) -> None:
    m = resumen_mape()
    mejora_prophet = (m["Prophet"] - m["LightGBM"]) / m["Prophet"] * 100
    mejora_sarimax = (m["SARIMAX"] - m["LightGBM"]) / m["SARIMAX"] * 100

    st.markdown(section_title("Conclusiones"), unsafe_allow_html=True)

    # ── Titular del resultado ───────────────────────────────────────────────────
    st.success(
        f"**Objetivo cumplido.** El modelo LightGBM alcanza un MAPE medio del "
        f"**{m['LightGBM']:.2f} %** en validación walk-forward, por debajo del umbral "
        f"objetivo del {OBJETIVO_MAPE:.0f} % fijado en el plan del TFM.",
        icon="🎯",
    )

    # ── KPIs de cierre ──────────────────────────────────────────────────────────
    kpi_row([
        ("Objetivo", f"< {OBJETIVO_MAPE:.0f} % MAPE", "plan del TFM"),
        ("Resultado", f"{m['LightGBM']:.2f} % MAPE", "LightGBM · walk-forward"),
        ("vs Prophet", f"−{mejora_prophet:.0f} %", "menos error"),
        ("vs SARIMAX", f"−{mejora_sarimax:.0f} %", "menos error"),
    ])
    st.markdown("<br>", unsafe_allow_html=True)

    # ── El recorrido ────────────────────────────────────────────────────────────
    st.markdown(section_title("El recorrido metodológico"), unsafe_allow_html=True)
    st.markdown(
        """
| Fase | Decisión clave | Justificación |
|------|----------------|---------------|
| **Datos** | REE (demanda) + Open-Meteo (clima) + INE (población) | Fuentes oficiales, gratuitas y horarias |
| **Ventana** | 2021 – 2026, sin 2020 | Evitar la distorsión del confinamiento (−15 % demanda) |
| **Preprocesado** | Suavizado del apagón y del 13-oct | Días anómalos que contaminaban target y lags |
| **Variables** | HDD/CDD en vez de temperatura cruda | Linealizan la relación en U con la demanda |
| **Variables** | Lags 24h y 168h | Los predictores más potentes (r = 0.81 y 0.90) |
| **Validación** | Walk-forward, 12 cortes mensuales | Sin fuga de datos; mide robustez en el tiempo |
| **Modelo** | LightGBM | Captura interacciones no lineales mejor que SARIMAX/Prophet |
"""
    )

    # ── Limitaciones ────────────────────────────────────────────────────────────
    st.markdown(section_title("Limitaciones"), unsafe_allow_html=True)
    st.markdown(
        """
- **Horizonte de 72 h.** El pronóstico operativo cubre 3 días; a mayor horizonte, el error crece.
- **Meteorología futura estimada.** Se usa el perfil histórico medio por mes y hora, no una previsión meteorológica real.
- **Sin variables económicas.** No se incluyen precio de la electricidad, actividad industrial ni festivos autonómicos.
- **Ámbito peninsular.** Excluye los sistemas insulares (Baleares y Canarias).
"""
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(
        "TFM · Máster en Data Science e IA · EBIS Business Techschool  |  "
        "Fuentes: REE e·sios · Open-Meteo (ERA5) · INE Padrón Municipal"
    )
