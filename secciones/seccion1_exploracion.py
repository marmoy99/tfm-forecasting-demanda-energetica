#!/usr/bin/env python3
"""
Sección 1 — Exploración del dataset
===================================
KPIs generales, serie histórica, perfil horario y estacionalidad mensual.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from theme import C_BLUE, C_AQUA, C_YELLOW, fig_base, section_title, show_fig


def render(df: pd.DataFrame) -> None:
    # ── KPIs ───────────────────────────────────────────────────────────────────
    st.markdown(section_title("Vista general del dataset"), unsafe_allow_html=True)

    periodo   = f"{df['datetime'].min().strftime('%b %Y')} – {df['datetime'].max().strftime('%b %Y')}"
    media_mw  = f"{df['demanda_mw'].mean():,.0f} MW"
    pico_mw   = f"{df['demanda_mw'].max():,.0f} MW"
    minimo_mw = f"{df['demanda_mw'].min():,.0f} MW"

    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, sub in [
        (c1, "Período", periodo,         f"{len(df):,} registros horarios"),
        (c2, "Demanda media", media_mw,  "promedio histórico"),
        (c3, "Demanda pico",  pico_mw,   "máximo registrado"),
        (c4, "Demanda mínima", minimo_mw, "mínimo registrado"),
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

    # ── Serie histórica ─────────────────────────────────────────────────────────
    st.markdown(section_title("Serie histórica de demanda (MW)"), unsafe_allow_html=True)

    gran_label = st.radio(
        "Granularidad",
        ["Horaria", "Diaria", "Semanal", "Mensual"],
        horizontal=True,
        index=2,
    )

    freq_map = {"Horaria": None, "Diaria": "D", "Semanal": "W", "Mensual": "ME"}
    freq = freq_map[gran_label]

    if freq:
        serie = (
            df.set_index("datetime")["demanda_mw"]
            .resample(freq)
            .mean()
            .reset_index()
        )
    else:
        serie = df[["datetime", "demanda_mw"]].copy()

    fig = fig_base(title=f"Demanda eléctrica peninsular · {gran_label.lower()}")
    fig.add_trace(go.Scatter(
        x=serie["datetime"],
        y=serie["demanda_mw"],
        mode="lines",
        line=dict(color=C_BLUE, width=2),
        name="Demanda MW",
        hovertemplate="<b>%{x|%d %b %Y %H:%M}</b><br>%{y:,.0f} MW<extra></extra>",
    ))
    fig.update_layout(showlegend=False, yaxis_title="MW", height=360)
    show_fig(fig)

    # ── Perfil horario ──────────────────────────────────────────────────────────
    st.markdown(section_title("Perfil horario por tipo de día"), unsafe_allow_html=True)
    st.caption("Demanda media (MW) por hora del día · separado por tipo de día")

    tipo_map = {
        "Laborable": (df["es_fin_de_semana"] == 0) & (df["es_festivo"] == 0),
        "Fin de semana": df["es_fin_de_semana"] == 1,
        "Festivo": df["es_festivo"] == 1,
    }
    colors = [C_BLUE, C_AQUA, C_YELLOW]

    fig2 = fig_base(title="Perfil horario de demanda")
    for (tipo, mask), color in zip(tipo_map.items(), colors):
        perfil = df[mask].groupby("hora")["demanda_mw"].mean().reset_index()
        fig2.add_trace(go.Scatter(
            x=perfil["hora"],
            y=perfil["demanda_mw"],
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=6, color=color),
            name=tipo,
            hovertemplate=f"<b>{tipo}</b> · Hora %{{x}}h<br>%{{y:,.0f}} MW<extra></extra>",
        ))
    fig2.update_layout(
        xaxis=dict(tickmode="linear", dtick=2, title="Hora del día"),
        yaxis_title="MW (media)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=340,
    )
    show_fig(fig2)

    # ── Estacionalidad mensual ──────────────────────────────────────────────────
    st.markdown(section_title("Estacionalidad mensual"), unsafe_allow_html=True)
    st.caption("Demanda media (MW) por mes · promedio de todos los años")

    meses_es = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
                "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
    mensual = df.groupby("mes")["demanda_mw"].mean().reset_index()
    mensual["mes_label"] = mensual["mes"].apply(lambda m: meses_es[m - 1])

    fig3 = fig_base(title="Estacionalidad mensual")
    fig3.add_trace(go.Bar(
        x=mensual["mes_label"],
        y=mensual["demanda_mw"],
        marker_color=C_BLUE,
        marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>%{y:,.0f} MW<extra></extra>",
    ))
    fig3.update_layout(
        yaxis_title="MW (media)",
        yaxis_range=[mensual["demanda_mw"].min() * 0.92, mensual["demanda_mw"].max() * 1.04],
        height=300,
        showlegend=False,
    )
    show_fig(fig3)
