#!/usr/bin/env python3
"""
dashboard.py — TFM Forecasting Demanda Eléctrica Peninsular
============================================================
Orquestador del dashboard. Cada sección vive en secciones/ y expone render(df).

Ejecutar: streamlit run dashboard.py
"""

import pandas as pd
import streamlit as st
from pathlib import Path

from theme import GLOBAL_CSS
from secciones import (seccion1_exploracion, seccion2_preprocesado,
                       seccion3_modelado, seccion4_prediccion, seccion5_conclusiones)

# ── Rutas ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA = ROOT / "data" / "processed" / "dataset_modelado.csv"

# ── Config página ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TFM · Demanda Eléctrica España",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ── Carga de datos (cacheada) ──────────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando dataset…")
def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


# ── Router de secciones ────────────────────────────────────────────────────────
SECCIONES = {
    "1 · Exploración del dataset": seccion1_exploracion.render,
    "2 · Preprocesado del dataset": seccion2_preprocesado.render,
    "3 · Modelado y validación": seccion3_modelado.render,
    "4 · Predicción final": seccion4_prediccion.render,
    "5 · Conclusiones": seccion5_conclusiones.render,
}


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚡ TFM · Demanda Eléctrica")
    st.markdown("**EBIS Business Techschool**")
    st.markdown("---")
    seleccion = st.radio("Sección", list(SECCIONES.keys()), label_visibility="collapsed")
    st.markdown("---")
    st.caption("Dataset: REE e·sios + Open-Meteo + INE  \n2021-01-01 → 2026-03-22")


# ══════════════════════════════════════════════════════════════════════════════
# CABECERA + CONTENIDO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## Forecasting de Demanda Eléctrica Peninsular")
st.markdown(
    "Predicción horaria de la demanda de la red eléctrica española a 7 días vista. "
    "Datos históricos del periodo **2021 – 2026** combinando consumo REE, "
    "meteorología Open-Meteo y demografía INE."
)

df = load_data()

# Ejecuta el render de la sección seleccionada
SECCIONES[seleccion](df)
