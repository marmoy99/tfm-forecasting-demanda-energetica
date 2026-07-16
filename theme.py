#!/usr/bin/env python3
"""
theme.py — Paleta, estilos y helpers de gráfico compartidos
============================================================
Paleta validada con el skill dataviz (colorblind-safe, light + dark).
Todos los módulos de sección importan de aquí.
"""

import plotly.graph_objects as go
import streamlit as st

# ── Paleta (dataviz skill — validated) ─────────────────────────────────────────
C_BLUE    = "#2a78d6"
C_AQUA    = "#1baf7a"
C_YELLOW  = "#eda100"
C_RED     = "#e34948"
C_SURFACE = "#fcfcfb"
C_GRID    = "#e1e0d9"
C_TEXT    = "#0b0b0b"
C_MUTED   = "#898781"

# ── CSS global (KPIs, títulos de sección, dark mode) ───────────────────────────
GLOBAL_CSS = """
<style>
  /* Fuente sistema, sin serif */
  html, body, [class*="css"] { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }

  /* Tarjetas KPI — light */
  .kpi-card {
    background: #f4f4f2;
    border-radius: 8px;
    padding: 18px 22px 14px;
    border-left: 3px solid #2a78d6;
  }
  .kpi-label { font-size: 0.78rem; color: #898781; font-weight: 500; letter-spacing: .04em; text-transform: uppercase; }
  .kpi-value { font-size: 1.85rem; font-weight: 700; color: #0b0b0b; line-height: 1.1; margin-top: 2px; }
  .kpi-sub   { font-size: 0.78rem; color: #52514e; margin-top: 3px; }

  /* Separador de sección — light */
  .section-title { font-size: 1.1rem; font-weight: 600; color: #0b0b0b; margin: 28px 0 4px; border-bottom: 1px solid #e1e0d9; padding-bottom: 6px; }

  /* Dark mode */
  @media (prefers-color-scheme: dark) {
    .kpi-card       { background: #2a2a28; }
    .kpi-value      { color: #ffffff; }
    .kpi-sub        { color: #c3c2b7; }
    .section-title  { color: #ffffff; border-bottom-color: #2c2c2a; }
  }
  /* Streamlit fuerza dark via atributo en el root */
  [data-theme="dark"] .kpi-card      { background: #2a2a28; }
  [data-theme="dark"] .kpi-value     { color: #ffffff; }
  [data-theme="dark"] .kpi-sub       { color: #c3c2b7; }
  [data-theme="dark"] .section-title { color: #ffffff; border-bottom-color: #2c2c2a; }
</style>
"""

# ── Layout base de Plotly ──────────────────────────────────────────────────────
PLOTLY_BASE = dict(
    paper_bgcolor=C_SURFACE,
    plot_bgcolor=C_SURFACE,
    font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=C_TEXT, size=12),
    margin=dict(l=10, r=10, t=36, b=10),
    xaxis=dict(showgrid=False, zeroline=False, linecolor=C_GRID, tickfont=dict(color=C_MUTED, size=11)),
    yaxis=dict(showgrid=True, gridcolor=C_GRID, zeroline=False, tickfont=dict(color=C_MUTED, size=11)),
    hoverlabel=dict(bgcolor="white", bordercolor=C_GRID, font_size=12, font_color=C_TEXT),
)


def fig_base(**overrides) -> go.Figure:
    """Devuelve una figura de Plotly con el layout base del TFM aplicado."""
    layout = {**PLOTLY_BASE, **overrides}
    return go.Figure(layout=go.Layout(**layout))


def section_title(text: str) -> str:
    """HTML de un título de sección (usar con st.markdown unsafe_allow_html=True)."""
    return f'<div class="section-title">{text}</div>'


def show_fig(fig: go.Figure, container=None) -> None:
    """
    Renderiza una figura respetando SIEMPRE nuestra paleta.

    theme=None evita que Streamlit sobreescriba los colores (en modo oscuro
    ponía el texto de la leyenda en blanco sobre nuestro fondo claro).
    """
    target = container if container is not None else st
    target.plotly_chart(fig, use_container_width=True, theme=None)
