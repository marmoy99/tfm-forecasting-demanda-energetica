#!/usr/bin/env python3
"""
Sección 3 — Modelado y validación walk-forward
==============================================
Compara los tres modelos (LightGBM, SARIMAX, Prophet) con validación
walk-forward sobre 12 cortes mensuales.
  1. Esquema de la validación walk-forward
  2. Comparativa de MAPE medio por modelo
  3. Evolución del MAPE corte a corte
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from theme import (C_BLUE, C_AQUA, C_YELLOW, C_GRID, C_MUTED,
                   fig_base, section_title, show_fig)

REPORTS = Path(__file__).resolve().parents[1] / "reports"

# Color fijo por modelo (se mantiene en todos los gráficos)
MODEL_COLOR = {
    "LightGBM": C_BLUE,
    "SARIMAX":  C_AQUA,
    "Prophet":  C_YELLOW,
}
OBJETIVO_MAPE = 3.0
SARIMAX_VARIANTE = "SARIMAX armónico + lag 168h"


@st.cache_data(show_spinner="Cargando resultados de modelos…")
def load_walk_forward() -> pd.DataFrame:
    """Une los 3 modelos en un único DataFrame: corte, modelo, mape."""
    lgb = pd.read_csv(REPORTS / "model_results" / "resultados_lightgbm_walk_forward.csv")
    lgb = lgb[["corte", "mape"]].assign(modelo="LightGBM")

    pro = pd.read_csv(REPORTS / "model_results" / "resultados_prophet_walk_forward.csv")
    pro = pro[["corte", "mape"]].assign(modelo="Prophet")

    sar = pd.read_csv(REPORTS / "model_comparison" / "resultados_sarimax_comparacion.csv",
                      encoding="utf-8-sig")
    sar = sar[sar["modelo"] == SARIMAX_VARIANTE][["corte", "mape"]].assign(modelo="SARIMAX")

    df = pd.concat([lgb, pro, sar], ignore_index=True)
    df["corte"] = pd.to_datetime(df["corte"])
    df["mape"] = df["mape"].astype(float)
    return df.sort_values(["modelo", "corte"]).reset_index(drop=True)


def render(df_dataset: pd.DataFrame) -> None:
    wf = load_walk_forward()
    resumen = wf.groupby("modelo")["mape"].mean().sort_values()
    mejor_modelo = resumen.index[0]
    mejor_mape = resumen.iloc[0]
    cortes = sorted(wf["corte"].unique())

    st.markdown(section_title("Modelado y validación walk-forward"), unsafe_allow_html=True)
    st.markdown(
        "Se comparan tres enfoques —**LightGBM**, **SARIMAX** y **Prophet**— con una "
        "validación temporal estricta: nunca se entrena con datos futuros. Cada modelo se "
        "evalúa sobre 12 cortes mensuales para medir su robustez a lo largo del año."
    )

    # ── KPIs ────────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, sub in [
        (c1, "Mejor modelo", mejor_modelo, "menor MAPE medio"),
        (c2, "MAPE medio", f"{mejor_mape:.2f} %", f"objetivo < {OBJETIVO_MAPE:.0f} %"),
        (c3, "Cortes evaluados", f"{len(cortes)}", "mensuales, walk-forward"),
        (c4, "Modelos comparados", "3", "+ baselines ingenuos"),
    ]:
        col.markdown(
            f'<div class="kpi-card">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-sub">{sub}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 1. ESQUEMA WALK-FORWARD
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(section_title("1 · Cómo funciona la validación walk-forward"), unsafe_allow_html=True)
    st.caption(
        "En cada corte se entrena con el histórico previo y se predice el tramo siguiente. "
        "Después el corte avanza un mes y se repite. Nunca se usa información del futuro."
    )

    st.markdown(
        '<div style="display:flex;gap:20px;margin-bottom:6px;font-size:0.82rem;">'
        f'<span><span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:{C_GRID};margin-right:5px;vertical-align:middle;"></span>Entrenamiento (histórico previo)</span>'
        f'<span><span style="display:inline-block;width:12px;height:12px;border-radius:2px;background:{C_BLUE};margin-right:5px;vertical-align:middle;"></span>Predicción evaluada</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    cortes_ts = [pd.Timestamp(c) for c in cortes]
    x_min = min(cortes_ts) - pd.Timedelta(days=395)
    x_max = max(cortes_ts) + pd.Timedelta(days=40)

    fig_wf = fig_base(title="Esquema de validación walk-forward · 12 cortes mensuales")
    for i, corte in enumerate(cortes_ts):
        train_start = corte - pd.Timedelta(days=365)
        test_end = corte + pd.Timedelta(days=3)
        # Barra de entrenamiento (gris)
        fig_wf.add_trace(go.Scatter(
            x=[train_start, corte], y=[i, i],
            mode="lines", line=dict(color=C_GRID, width=13),
            showlegend=False,
            hovertemplate=f"<b>Corte {corte:%b %Y}</b><br>Entrenamiento<extra></extra>",
        ))
        # Barra de predicción evaluada (azul)
        fig_wf.add_trace(go.Scatter(
            x=[corte, test_end], y=[i, i],
            mode="lines", line=dict(color=C_BLUE, width=13),
            showlegend=False,
            hovertemplate=f"<b>Corte {corte:%b %Y}</b><br>Predicción evaluada<extra></extra>",
        ))
    fig_wf.update_layout(
        xaxis=dict(title=None, range=[x_min, x_max], tickformat="%b %Y",
                   nticks=8, tickangle=0),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(cortes_ts))),
            ticktext=[c.strftime("%b %Y") for c in cortes_ts],
            range=[len(cortes_ts) - 0.5, -0.5],  # primer corte arriba
            showgrid=False,
        ),
        height=460,
        margin=dict(l=95, r=55, t=50, b=60),
    )
    show_fig(fig_wf)

    # ══════════════════════════════════════════════════════════════════════════
    # 2. MAPE MEDIO POR MODELO
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(section_title("2 · Comparativa: MAPE medio por modelo"), unsafe_allow_html=True)
    st.caption("Error porcentual medio absoluto sobre los 12 cortes · menor es mejor")

    fig_bar = fig_base(title="MAPE medio por modelo (12 cortes)")
    fig_bar.add_trace(go.Bar(
        x=resumen.values,
        y=resumen.index,
        orientation="h",
        marker_color=[MODEL_COLOR[m] for m in resumen.index],
        marker_line_width=0,
        text=[f"{v:.2f} %" for v in resumen.values],
        textposition="outside",
        textfont=dict(color=C_MUTED),
        hovertemplate="<b>%{y}</b><br>MAPE medio = %{x:.2f} %<extra></extra>",
    ))
    fig_bar.add_vline(x=OBJETIVO_MAPE, line_width=1.5, line_dash="dash", line_color=C_MUTED,
                      annotation_text=f"objetivo {OBJETIVO_MAPE:.0f} %",
                      annotation_font_color=C_MUTED, annotation_position="top")
    fig_bar.update_layout(
        xaxis=dict(title="MAPE medio (%)", range=[0, max(resumen.values) * 1.25]),
        yaxis=dict(title=None, autorange="reversed"),
        height=280,
        showlegend=False,
    )
    show_fig(fig_bar)

    st.success(
        f"**{mejor_modelo} es el único modelo que cumple el objetivo del TFM** "
        f"(MAPE {mejor_mape:.2f} % < {OBJETIVO_MAPE:.0f} %). Los árboles de gradiente capturan "
        "las interacciones no lineales entre calendario, meteorología y lags mejor que "
        "SARIMAX (lineal) o Prophet (aditivo por componentes).",
        icon="🏆",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 3. MAPE CORTE A CORTE
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(section_title("3 · Evolución del MAPE corte a corte"), unsafe_allow_html=True)
    st.caption("MAPE en cada uno de los 12 cortes · muestra la estabilidad de cada modelo")

    fig_line = fig_base(title="MAPE por corte temporal")
    for modelo in ["LightGBM", "SARIMAX", "Prophet"]:
        sub = wf[wf["modelo"] == modelo].sort_values("corte")
        fig_line.add_trace(go.Scatter(
            x=sub["corte"], y=sub["mape"],
            mode="lines+markers",
            line=dict(color=MODEL_COLOR[modelo], width=2),
            marker=dict(size=6, color=MODEL_COLOR[modelo]),
            name=modelo,
            hovertemplate=f"<b>{modelo}</b> · %{{x|%b %Y}}<br>MAPE = %{{y:.2f}} %<extra></extra>",
        ))
    fig_line.add_hline(y=OBJETIVO_MAPE, line_width=1.5, line_dash="dash", line_color=C_MUTED,
                       annotation_text=f"objetivo {OBJETIVO_MAPE:.0f} %",
                       annotation_font_color=C_MUTED, annotation_position="top left")
    fig_line.update_layout(
        xaxis=dict(title=None),
        yaxis=dict(title="MAPE (%)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=360,
    )
    show_fig(fig_line)

    st.info(
        "LightGBM se mantiene por debajo o cerca del 3 % en casi todos los cortes, mientras "
        "que Prophet es sistemáticamente el peor. El pico de septiembre 2025 afecta a todos "
        "los modelos (transición térmica de verano a otoño, difícil de anticipar). Esta vista "
        "confirma que la ventaja de LightGBM no es fruto de un único corte afortunado.",
        icon="📈",
    )
