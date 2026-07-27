#!/usr/bin/env python3
"""
Sección 4 — Predicción final
============================
Pronóstico a 72 h (2026-03-23 → 2026-03-25) de los tres modelos, encadenado
con el histórico real, y el modelo seleccionado con su banda de error esperada.
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from theme import (C_BLUE, C_AQUA, C_YELLOW, C_MUTED, C_GRID,
                   fig_base, section_title, show_fig, kpi_row)

REPORTS = Path(__file__).resolve().parents[1] / "reports"
C_REAL = "#3d3d3a"  # gris oscuro para el histórico real

MODEL_COLOR = {
    "LightGBM": C_BLUE,
    "SARIMAX":  C_AQUA,
    "Prophet":  C_YELLOW,
}
PRED_FILES = {
    "LightGBM": REPORTS / "predictions" / "prediccion_lightgbm_final.csv",
    "SARIMAX":  REPORTS / "predictions" / "prediccion_sarimax_armonico_final.csv",
    "Prophet":  REPORTS / "predictions" / "prediccion_prophet_final.csv",
}


@st.cache_data(show_spinner="Cargando predicciones finales…")
def load_predictions() -> pd.DataFrame:
    frames = []
    for name, path in PRED_FILES.items():
        d = pd.read_csv(path, usecols=["datetime", "prediccion_demanda_mw"], encoding="utf-8-sig")
        d["datetime"] = pd.to_datetime(d["datetime"])
        d = d.rename(columns={"prediccion_demanda_mw": "prediccion"})
        d["modelo"] = name
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def mape_lightgbm() -> float:
    lgb = pd.read_csv(REPORTS / "model_results" / "resultados_lightgbm_walk_forward.csv")
    return float(lgb["mape"].mean())


def render(df_dataset: pd.DataFrame) -> None:
    preds = load_predictions()
    mape_lgb = mape_lightgbm()

    inicio = preds["datetime"].min()
    fin = preds["datetime"].max()
    horas = preds["datetime"].nunique()

    st.markdown(section_title("Predicción final a 72 horas"), unsafe_allow_html=True)
    st.markdown(
        "Con los modelos ya validados, se genera el pronóstico operativo: la demanda "
        "horaria de los **3 días siguientes** al último dato disponible. Las variables "
        "meteorológicas futuras se estiman con el perfil histórico medio de cada mes y hora."
    )

    # ── KPIs ────────────────────────────────────────────────────────────────────
    kpi_row([
        ("Horizonte", f"{horas} h", "3 días horarios"),
        ("Inicio", inicio.strftime("%d %b %Y"), "tras el último dato"),
        ("Modelo operativo", "LightGBM", f"MAPE esperado {mape_lgb:.2f} %"),
        ("Modelos comparados", "3", "LightGBM · SARIMAX · Prophet"),
    ])
    st.markdown("<br>", unsafe_allow_html=True)

    # Demanda real: 5 días previos + el propio periodo del pronóstico,
    # para poder comparar predicción vs. realidad sobre las mismas fechas.
    hist = df_dataset[(df_dataset["datetime"] >= inicio - pd.Timedelta(days=5)) &
                      (df_dataset["datetime"] <= fin)][["datetime", "demanda_mw"]]
    # Punto real inmediatamente anterior al pronóstico (para encadenar sin salto)
    ultimo_real = df_dataset[df_dataset["datetime"] < inicio].iloc[-1]

    # ══════════════════════════════════════════════════════════════════════════
    # 1. TRASPASO HISTÓRICO → FORECAST (3 modelos)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(section_title("1 · Del histórico real al pronóstico"), unsafe_allow_html=True)
    st.caption("Demanda real (histórico + periodo pronosticado) vs. pronóstico a 72 h de los tres modelos")

    fig1 = fig_base(title="Demanda real y predicción a 72 h")
    fig1.add_trace(go.Scatter(
        x=hist["datetime"], y=hist["demanda_mw"],
        mode="lines", line=dict(color=C_REAL, width=2),
        name="Demanda real",
        hovertemplate="<b>Real</b> %{x|%d %b %H:%M}<br>%{y:,.0f} MW<extra></extra>",
    ))
    # Línea vertical marcando el inicio del pronóstico
    fig1.add_vline(x=inicio, line_width=1, line_dash="dot", line_color=C_MUTED,
                   annotation_text="inicio pronóstico", annotation_font_color=C_MUTED,
                   annotation_position="top left")
    for modelo in ["LightGBM", "SARIMAX", "Prophet"]:
        sub = preds[preds["modelo"] == modelo].sort_values("datetime")
        # Encadena el último punto real con la primera predicción
        xs = [ultimo_real["datetime"]] + sub["datetime"].tolist()
        ys = [ultimo_real["demanda_mw"]] + sub["prediccion"].tolist()
        fig1.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color=MODEL_COLOR[modelo], width=2,
                      dash="solid" if modelo == "LightGBM" else "dot"),
            name=f"Pred. {modelo}",
            hovertemplate=f"<b>{modelo}</b> %{{x|%d %b %H:%M}}<br>%{{y:,.0f}} MW<extra></extra>",
        ))
    fig1.update_layout(
        yaxis_title="MW",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=380,
    )
    show_fig(fig1)

    st.info(
        "Los tres modelos coinciden en el arranque y divergen a medida que se alejan del "
        "último dato conocido. LightGBM (línea sólida) reproduce con más fidelidad los dobles "
        "picos diarios; Prophet tiende a suavizar y SARIMAX queda en un punto intermedio.",
        icon="🔮",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 2. MODELO SELECCIONADO CON BANDA DE ERROR
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(section_title("2 · Modelo operativo: LightGBM con banda de error"), unsafe_allow_html=True)
    st.caption(f"Pronóstico de LightGBM con la banda ±{mape_lgb:.2f} % (MAPE medio del walk-forward)")

    lgb = preds[preds["modelo"] == "LightGBM"].sort_values("datetime")
    factor = mape_lgb / 100.0
    upper = lgb["prediccion"] * (1 + factor)
    lower = lgb["prediccion"] * (1 - factor)

    fig2 = fig_base(title="Predicción LightGBM y banda de error esperada")
    # Banda (relleno)
    fig2.add_trace(go.Scatter(
        x=lgb["datetime"].tolist() + lgb["datetime"].tolist()[::-1],
        y=upper.tolist() + lower.tolist()[::-1],
        fill="toself", fillcolor="rgba(42,120,214,0.15)",
        line=dict(width=0), hoverinfo="skip",
        name=f"Banda ±{mape_lgb:.2f} %",
    ))
    fig2.add_trace(go.Scatter(
        x=lgb["datetime"], y=lgb["prediccion"],
        mode="lines", line=dict(color=C_BLUE, width=2.5),
        name="Predicción LightGBM",
        hovertemplate="<b>LightGBM</b> %{x|%d %b %H:%M}<br>%{y:,.0f} MW<extra></extra>",
    ))
    fig2.update_layout(
        yaxis_title="MW",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        height=360,
    )
    show_fig(fig2)

    st.success(
        f"La banda representa el margen de error esperable según la validación: con un MAPE "
        f"medio del {mape_lgb:.2f} %, la demanda real debería caer dentro de la franja azul en "
        "la mayoría de las horas. Es el pronóstico que se entregaría al operador de red.",
        icon="✅",
    )

    # ══════════════════════════════════════════════════════════════════════════
    # 3. TABLA DESCARGABLE
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown(section_title("3 · Valores del pronóstico"), unsafe_allow_html=True)

    tabla = preds.pivot_table(index="datetime", columns="modelo", values="prediccion").round(0)
    tabla = tabla[["LightGBM", "SARIMAX", "Prophet"]].reset_index()
    tabla["datetime"] = tabla["datetime"].dt.strftime("%Y-%m-%d %H:%M")

    st.dataframe(tabla, use_container_width=True, height=280, hide_index=True)
    st.download_button(
        "Descargar pronóstico (CSV)",
        tabla.to_csv(index=False).encode("utf-8-sig"),
        file_name="prediccion_final_72h.csv",
        mime="text/csv",
    )
