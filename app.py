# -*- coding: utf-8 -*-
"""
SIAAP — Sistema Inteligente de Apoyo al Aprendizaje Personalizado
App de Streamlit que expone el modelo de Machine Learning supervisado
(clasificación multiclase) desarrollado en el notebook CRISP-ML(Q).

Ejecutar localmente:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

RANDOM_STATE = 42

# ----------------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="SIAAP · Apoyo al Aprendizaje Personalizado",
    page_icon="🎓",
    layout="wide",
)

PRIMARY = "#2B4C7E"
GREEN = "#2F8F5B"
AMBER = "#E8A33D"
RED = "#C0392B"

st.markdown(
    """
    <style>
    .main {background-color: #EEF1F6;}
    .stMetric {background: white; padding: 14px; border-radius: 12px; border: 1px solid #D6DCE6;}
    h1, h2, h3 {color: #16233A;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# 1) DATOS SINTÉTICOS + 2) ENTRENAMIENTO DEL MODELO (cacheado)
#    Se entrena en el arranque de la app para que el repositorio no
#    dependa de un archivo binario de modelo (evita problemas de
#    compatibilidad de versiones entre entornos de despliegue).
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Entrenando el modelo (CRISP-ML · Fase 3)...")
def entrenar_modelo():
    rng = np.random.default_rng(RANDOM_STATE)
    n = 1200

    horas_estudio_semanal = np.clip(rng.normal(10, 4, n), 0, 30)
    asistencia_pct = np.clip(rng.normal(80, 12, n), 30, 100)
    promedio_anterior = np.clip(rng.normal(6.5, 1.4, n), 0, 10)
    participacion_clase = np.clip(rng.normal(6, 2, n), 0, 10)
    uso_plataforma_horas = np.clip(rng.normal(5, 2.5, n), 0, 20)
    entregas_a_tiempo_pct = np.clip(rng.normal(75, 15, n), 0, 100)
    horas_sueno = np.clip(rng.normal(7, 1.3, n), 3, 11)
    apoyo_familiar = rng.choice(["Bajo", "Medio", "Alto"], size=n, p=[0.25, 0.45, 0.30])

    apoyo_map = {"Bajo": 0, "Medio": 0.5, "Alto": 1}
    apoyo_num = np.array([apoyo_map[a] for a in apoyo_familiar])

    score_real = (
        0.28 * (horas_estudio_semanal / 30)
        + 0.22 * (asistencia_pct / 100)
        + 0.20 * (promedio_anterior / 10)
        + 0.10 * (participacion_clase / 10)
        + 0.08 * (uso_plataforma_horas / 20)
        + 0.07 * (entregas_a_tiempo_pct / 100)
        + 0.03 * (horas_sueno / 11)
        + 0.10 * apoyo_num
        + rng.normal(0, 0.02, n)
    )

    df = pd.DataFrame({
        "horas_estudio_semanal": horas_estudio_semanal,
        "asistencia_pct": asistencia_pct,
        "promedio_anterior": promedio_anterior,
        "participacion_clase": participacion_clase,
        "uso_plataforma_horas": uso_plataforma_horas,
        "entregas_a_tiempo_pct": entregas_a_tiempo_pct,
        "horas_sueno": horas_sueno,
        "apoyo_familiar": apoyo_familiar,
    })

    q1, q2 = np.quantile(score_real, [0.33, 0.66])
    rendimiento = np.where(score_real <= q1, "Bajo",
                    np.where(score_real <= q2, "Medio", "Alto"))
    df["rendimiento"] = rendimiento

    num_features = ["horas_estudio_semanal", "asistencia_pct", "promedio_anterior",
                     "participacion_clase", "uso_plataforma_horas",
                     "entregas_a_tiempo_pct", "horas_sueno"]
    cat_features = ["apoyo_familiar"]

    X = df.drop(columns=["rendimiento"])
    y = df["rendimiento"]

    preprocesador = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(drop="first"), cat_features),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = Pipeline([
        ("preprocesador", preprocesador),
        ("modelo", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metricas = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
    }

    return pipeline, metricas


def recomendar_apoyo(pipeline, datos_estudiante: dict) -> dict:
    entrada = pd.DataFrame([datos_estudiante])
    prediccion = pipeline.predict(entrada)[0]
    probabilidades = pipeline.predict_proba(entrada)[0]
    clases = pipeline.classes_
    confianza = dict(zip(clases, probabilidades))

    recomendaciones = {
        "Bajo": "🔴 Prioridad alta: tutoría personalizada, plan de estudio guiado y "
                "seguimiento semanal del docente.",
        "Medio": "🟡 Seguimiento moderado: técnicas de estudio, recordatorios de "
                 "entregas y material de refuerzo opcional.",
        "Alto": "🟢 Buen camino: retos de profundización y contenido avanzado opcional.",
    }
    return {
        "prediccion": prediccion,
        "confianza": confianza,
        "recomendacion": recomendaciones[prediccion],
    }


pipeline, metricas = entrenar_modelo()

# ----------------------------------------------------------------------
# ENCABEZADO
# ----------------------------------------------------------------------
st.title("🎓 SIAAP — Sistema Inteligente de Apoyo al Aprendizaje Personalizado")
st.caption(
    "Modelo de Machine Learning supervisado (clasificación multiclase) desarrollado "
    "con metodología **CRISP-ML(Q)**. Datos sintéticos con fines demostrativos."
)

col_a, col_b, col_c = st.columns(3)
col_a.metric("F1-score macro (test)", f"{metricas['f1_macro']:.2f}")
col_b.metric("Accuracy (test)", f"{metricas['accuracy']:.2f}")
col_c.metric("Clases predichas", "Bajo · Medio · Alto")

st.divider()

# ----------------------------------------------------------------------
# FORMULARIO DE ENTRADA
# ----------------------------------------------------------------------
st.subheader("Simula el perfil de un estudiante")

col1, col2 = st.columns(2)

with col1:
    horas_estudio_semanal = st.slider("Horas de estudio semanal", 0.0, 30.0, 10.0, 0.5)
    asistencia_pct = st.slider("Asistencia (%)", 30.0, 100.0, 80.0, 1.0)
    promedio_anterior = st.slider("Promedio anterior (0–10)", 0.0, 10.0, 6.5, 0.1)
    participacion_clase = st.slider("Participación en clase (0–10)", 0.0, 10.0, 6.0, 0.5)

with col2:
    uso_plataforma_horas = st.slider("Uso semanal de la plataforma (horas)", 0.0, 20.0, 5.0, 0.5)
    entregas_a_tiempo_pct = st.slider("Entregas a tiempo (%)", 0.0, 100.0, 75.0, 1.0)
    horas_sueno = st.slider("Horas de sueño diarias", 3.0, 11.0, 7.0, 0.5)
    apoyo_familiar = st.selectbox("Apoyo familiar percibido", ["Bajo", "Medio", "Alto"], index=1)

if st.button("🔮 Predecir rendimiento y generar recomendación", type="primary"):
    datos_estudiante = {
        "horas_estudio_semanal": horas_estudio_semanal,
        "asistencia_pct": asistencia_pct,
        "promedio_anterior": promedio_anterior,
        "participacion_clase": participacion_clase,
        "uso_plataforma_horas": uso_plataforma_horas,
        "entregas_a_tiempo_pct": entregas_a_tiempo_pct,
        "horas_sueno": horas_sueno,
        "apoyo_familiar": apoyo_familiar,
    }

    resultado = recomendar_apoyo(pipeline, datos_estudiante)
    color_map = {"Bajo": RED, "Medio": AMBER, "Alto": GREEN}
    color = color_map[resultado["prediccion"]]

    st.markdown(f"### Predicción: <span style='color:{color}'>{resultado['prediccion']}</span>",
                unsafe_allow_html=True)
    st.info(resultado["recomendacion"])

    fig = go.Figure(go.Bar(
        x=list(resultado["confianza"].values()),
        y=list(resultado["confianza"].keys()),
        orientation="h",
        marker_color=[color_map[c] for c in resultado["confianza"].keys()],
    ))
    fig.update_layout(
        title="Confianza del modelo por clase",
        xaxis_title="Probabilidad",
        height=300,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption(
    "Proyecto académico de Machine Learning supervisado · Metodología CRISP-ML(Q) · "
    "Notebook completo disponible en `/notebook` del repositorio."
)
