#!/usr/bin/env python3
"""
Sección 2 — Preprocesado y construcción del dataset
===================================================
Muestra las transformaciones aplicadas a los datos crudos:
  1. Suavizado de outliers (apagón 28-abr-2025 y 13-oct-2025)
  2. Transformación temperatura → HDD / CDD
  3. Creación de los lags de demanda (24h y 168h)
"""

from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from theme import C_BLUE, C_AQUA, C_YELLOW, C_RED, C_MUTED, fig_base, section_title, show_fig

RAW_DEMAND = Path(__file__).resolve().parents[1] / "data" / "raw" / "demanda_peninsular_horaria.csv"

OUTLIER_DATES = {
    "Apagón · 28 abr 2025": "2025-04-28",
    "Anomalía · 13 oct 2025": "2025-10-13",
}


@st.cache_data(show_spinner="Cargando demanda cruda…")
def load_raw_demand() -> pd.DataFrame:
    df = pd.read_csv(RAW_DEMAND, usecols=["datetime", "demanda_mw"], encoding="utf-8-sig")
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def render(df: pd.DataFrame) -> None:
    st.markdown(section_title("De los datos crudos al dataset de modelado"), unsafe_allow_html=True)
    st.markdown(
        "Antes de entrenar, los datos pasan por tres transformaciones clave. "
        "Cada una responde a un problema concreto detectado en la fase de exploración."
    )

    raw = load_raw_demand()

    # ══════════════════════════════════════════════════════════════════════════
    # 1. SUAVIZADO DE OUTLIERS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(section_title("1 · Suavizado de días anómalos"), unsafe_allow_html=True)
    st.caption(
        "Dos días con demanda no representativa distorsionaban el histórico. "
        "Se sustituyen por la media de la misma hora ±7 días (mismo día de la semana)."
    )

    cols = st.columns(2)
    for (titulo, fecha_str), col in zip(OUTLIER_DATES.items(), cols):
        fecha = pd.Timestamp(fecha_str)
        ini, fin = fecha - pd.Timedelta(days=3), fecha + pd.Timedelta(days=4)

        r = raw[(raw["datetime"] >= ini) & (raw["datetime"] < fin)]
        s = df[(df["datetime"] >= ini) & (df["datetime"] < fin)]

        fig = fig_base(title=titulo)
        fig.add_trace(go.Scatter(
            x=r["datetime"], y=r["demanda_mw"], mode="lines",
            line=dict(color=C_RED, width=2),
            name="Original (crudo)",
            hovertemplate="<b>Crudo</b> %{x|%d %b %H:%M}<br>%{y:,.0f} MW<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=s["datetime"], y=s["demanda_mw"], mode="lines",
            line=dict(color=C_BLUE, width=2, dash="dot"),
            name="Suavizado",
            hovertemplate="<b>Suavizado</b> %{x|%d %b %H:%M}<br>%{y:,.0f} MW<extra></extra>",
        ))
        # Sombreado del día anómalo
        fig.add_vrect(
            x0=fecha, x1=fecha + pd.Timedelta(days=1),
            fillcolor=C_MUTED, opacity=0.10, line_width=0,
        )
        fig.update_layout(
            yaxis_title="MW",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            height=300,
        )
        show_fig(fig, col)

    st.info(
        "El **28 de abril de 2025** el apagón peninsular colapsó la demanda registrada; "
        "el **13 de octubre de 2025** hubo otra anomalía puntual. Sin corregir, estos días "
        "contaminarían tanto el target como los lags (`lag_168h` los arrastraría una semana "
        "después). Tras el suavizado, la correlación de los lags con la demanda **subió** "
        "(lag_168h: 0.877 → 0.90).",
        icon="🩹",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 2. TEMPERATURA NACIONAL PONDERADA POR POBLACIÓN
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(section_title("2 · Temperatura nacional ponderada por población"), unsafe_allow_html=True)
    st.caption(
        "La variable t_nac no es la temperatura de una ciudad, sino la media de 7 capitales "
        "peninsulares ponderada por la población de su provincia (Padrón INE, valores aprox.)."
    )

    # Población provincial (millones, Padrón INE — aproximado, uso ilustrativo)
    poblacion = {
        "Madrid": 6.90, "Barcelona": 5.80, "Valencia": 2.66, "Sevilla": 1.95,
        "Málaga": 1.75, "Bilbao": 1.15, "Zaragoza": 0.98,
    }
    pob = pd.Series(poblacion).sort_values()
    pesos = pob / pob.sum() * 100

    fig_pesos = fig_base(title="Peso de cada capital en la temperatura nacional")
    fig_pesos.add_trace(go.Bar(
        x=pesos.values, y=pesos.index, orientation="h",
        marker_color=C_BLUE, marker_line_width=0,
        text=[f"{v:.0f} %" for v in pesos.values], textposition="outside",
        textfont=dict(color=C_MUTED),
        hovertemplate="<b>%{y}</b><br>peso = %{x:.1f} %<extra></extra>",
    ))
    fig_pesos.update_layout(
        xaxis=dict(title="Peso en la media (%)", range=[0, max(pesos.values) * 1.25]),
        yaxis=dict(title=None),
        height=300, showlegend=False,
    )
    show_fig(fig_pesos)

    st.info(
        "En lugar de una media geográfica simple (todas las ciudades pesando igual), cada "
        "capital pondera según la población de su provincia. Así la temperatura refleja el "
        "clima que experimenta **la mayoría de la población** —y por tanto la mayor parte de "
        "la demanda—. Madrid y Barcelona concentran cerca del **60 %** del peso. Las 7 ciudades "
        "cubren el arco climático peninsular: interior (Madrid, Zaragoza), Mediterráneo "
        "(Barcelona, Valencia, Málaga), sur (Sevilla) y norte atlántico (Bilbao).",
        icon="⚖️",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 3. TEMPERATURA → HDD / CDD
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(section_title("3 · Ingeniería de variables: temperatura → HDD / CDD"), unsafe_allow_html=True)
    st.caption(
        "La temperatura cruda tiene una relación en U con la demanda. Se descompone en dos "
        "variables lineales usando el umbral de confort de 18 °C."
    )

    # Curva determinista a partir del rango real de t_nac
    t = np.linspace(df["t_nac"].min(), df["t_nac"].max(), 200)
    hdd = np.maximum(0.0, 18.0 - t)
    cdd = np.maximum(0.0, t - 18.0)

    fig_hdd = fig_base(title="HDD y CDD en función de la temperatura nacional")
    fig_hdd.add_trace(go.Scatter(
        x=t, y=hdd, mode="lines", line=dict(color=C_BLUE, width=2.5),
        name="HDD = max(0, 18 − T)",
        hovertemplate="T = %{x:.1f} °C<br>HDD = %{y:.1f}<extra></extra>",
    ))
    fig_hdd.add_trace(go.Scatter(
        x=t, y=cdd, mode="lines", line=dict(color=C_YELLOW, width=2.5),
        name="CDD = max(0, T − 18)",
        hovertemplate="T = %{x:.1f} °C<br>CDD = %{y:.1f}<extra></extra>",
    ))
    fig_hdd.add_vline(x=18, line_width=1, line_dash="dot", line_color=C_MUTED,
                      annotation_text="18 °C (confort)", annotation_font_color=C_MUTED,
                      annotation_position="top right")
    fig_hdd.update_layout(
        xaxis_title="Temperatura nacional ponderada (°C)",
        yaxis_title="Grados-día",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=340,
    )
    show_fig(fig_hdd)

    st.info(
        "En lugar de dar al modelo la temperatura cruda (relación no lineal, difícil de "
        "aprender para modelos lineales), se separa en **HDD** (cuánto frío hay que compensar "
        "con calefacción) y **CDD** (cuánto calor con refrigeración). Cada una es lineal y "
        "el modelo las interpreta directamente. Por debajo de 18 °C solo actúa HDD; por "
        "encima, solo CDD.",
        icon="🌡️",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 3. CREACIÓN DE LAGS
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(section_title("4 · Creación de los lags de demanda"), unsafe_allow_html=True)
    st.caption(
        "Un lag es la demanda desplazada en el tiempo: para cada hora se añade lo que se "
        "consumió 24 h y 168 h (1 semana) antes. Son los predictores más potentes."
    )

    win_ini = pd.Timestamp("2024-02-12")
    win_fin = pd.Timestamp("2024-02-22")
    w = df[(df["datetime"] >= win_ini) & (df["datetime"] < win_fin)]

    fig_lag = fig_base(title="Demanda y sus lags · feb 2024")
    fig_lag.add_trace(go.Scatter(
        x=w["datetime"], y=w["demanda_mw"], mode="lines",
        line=dict(color=C_BLUE, width=2.5),
        name="Demanda (t)",
        hovertemplate="<b>Demanda</b> %{x|%d %b %H:%M}<br>%{y:,.0f} MW<extra></extra>",
    ))
    fig_lag.add_trace(go.Scatter(
        x=w["datetime"], y=w["demanda_lag_24h"], mode="lines",
        line=dict(color=C_AQUA, width=1.5),
        name="lag 24h (ayer)",
        hovertemplate="<b>lag 24h</b> %{x|%d %b %H:%M}<br>%{y:,.0f} MW<extra></extra>",
    ))
    fig_lag.add_trace(go.Scatter(
        x=w["datetime"], y=w["demanda_lag_168h"], mode="lines",
        line=dict(color=C_YELLOW, width=1.5, dash="dot"),
        name="lag 168h (semana pasada)",
        hovertemplate="<b>lag 168h</b> %{x|%d %b %H:%M}<br>%{y:,.0f} MW<extra></extra>",
    ))
    fig_lag.update_layout(
        yaxis_title="MW",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=340,
    )
    show_fig(fig_lag)

    st.info(
        "El **lag 24h** reproduce el perfil del día anterior; el **lag 168h** el de la semana "
        "pasada, alineando festivos y fines de semana con su equivalente. En el gráfico se ve "
        "cómo las tres curvas comparten forma, con el lag 168h como el más fiel. "
        "Las primeras 168 horas del dataset quedan sin lag (NaN) y se descartan al entrenar.",
        icon="🔁",
    )
