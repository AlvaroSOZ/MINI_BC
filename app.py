# ============ 1. PAQUETES Y CONFIGURACIÓN ============
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.filters.hp_filter import hpfilter
from datetime import datetime
import plotly.graph_objects as go
import random

# Configuración general de la página
st.set_page_config(page_title="Mini Banco Central", layout="wide")
st.title("🏦 Mini Banco Central")

# ============ 2. CARGA Y PREPARACIÓN DE DATOS ============
@st.cache_data(show_spinner=False)
def cargar_datos():
    df = pd.read_csv("SIMULACION/DATOS.csv")

    if "pbi" not in df.columns:
        st.error("El archivo no contiene la columna 'pbi' necesaria para calcular la brecha.")
        st.stop()

    df["fecha"] = pd.date_range(start="2003-09-01", periods=len(df), freq="MS")

    ciclo, tendencia = hpfilter(df["pbi"], lamb=14400)
    df["brecha"] = (df["pbi"] - tendencia) / tendencia * 100

    return df

df = cargar_datos()

# ============ 3. CONSTANTES DEL MODELO ============
PI_META = 3.0      # Meta de inflación
R = 2.5            # Tasa de interés neutral o libre de riesgo
START_INDEX = 256  # Enero 2025 (desde sep-2003 son 256 meses)
ERROR_RANGE = (-0.15, 0.15)  # Rango de errores aleatorios para simulación

@st.cache_data
def preparar_variables(df):
    df["infla_t_1"] = df["infla"].shift(1)
    df["infla_t_2"] = df["infla"].shift(2)
    df["einfla_t_1"] = df["einfla"].shift(1)
    df["brecha_t_1"] = df["brecha"].shift(1)
    df["tasa_t_1"] = df["tasa"].shift(1)
    df["tasa_t_2"] = df["tasa"].shift(2)
    df["gap_real"] = df["tasa_t_1"] - df["einfla_t_1"] - R
    df["gap_einfla"] = df["einfla_t_1"] - PI_META
    df["r_pi_meta"] = R + PI_META
    df["delta_einfla"] = df["einfla"] - df["einfla"].shift(12)

    if "epbi" not in df.columns:
        df["epbi"] = df["brecha_t_1"]  # Valor inicial para epbi

    return df.dropna()

df = preparar_variables(df)

@st.cache_data
def estimar_coeficientes(df):
    # REGRESIÓN 1: INFLACIÓN
    X_pi = sm.add_constant(df[["infla_t_1", "infla_t_2", "einfla_t_1", "brecha_t_1"]])
    y_pi = df["infla"]
    modelo_pi = sm.OLS(y_pi, X_pi).fit()
    a1 = modelo_pi.params.get("infla_t_1", 0)
    a2 = modelo_pi.params.get("infla_t_2", 0)
    a3 = modelo_pi.params.get("einfla_t_1", 0)
    a4 = modelo_pi.params.get("brecha_t_1", 0)

    # REGRESIÓN 2: BRECHA DEL PRODUCTO
    X_y = sm.add_constant(df[["brecha_t_1", "epbi", "gap_real", "delta_einfla"]])
    y_y = df["brecha"]
    modelo_y = sm.OLS(y_y, X_y).fit()
    b1 = modelo_y.params.get("brecha_t_1", 0)
    b2 = modelo_y.params.get("epbi", 0)
    b3 = modelo_y.params.get("gap_real", 0)
    b4 = modelo_y.params.get("delta_einfla", 0)

    # REGRESIÓN 3: TASA DE INTERÉS
    X_i = sm.add_constant(df[["tasa_t_1", "tasa_t_2", "r_pi_meta", "gap_einfla", "brecha_t_1"]])
    y_i = df["tasa"]
    modelo_i = sm.OLS(y_i, X_i).fit()
    c1 = modelo_i.params.get("tasa_t_1", 0)
    c2 = modelo_i.params.get("tasa_t_2", 0)
    c3 = modelo_i.params.get("r_pi_meta", 0)
    c4 = modelo_i.params.get("gap_einfla", 0)
    c5 = modelo_i.params.get("brecha_t_1", 0)

    return (a1, a2, a3, a4), (b1, b2, b3, b4), (c1, c2, c3, c4, c5)

a_params, b_params, c_params = estimar_coeficientes(df)

# ============ 5. INICIALIZACIÓN DEL JUEGO Y VARIABLES DE SESIÓN ============
if "index" not in st.session_state:
    st.session_state.index = START_INDEX

if "df_juego" not in st.session_state:
    st.session_state.df_juego = df.copy()

if "tasa_input" not in st.session_state:
    st.session_state.tasa_input = R + PI_META  # valor inicial razonable

if "perdio" not in st.session_state:
    st.session_state.perdio = False

# ============ 6. LAYOUT DE INTERFAZ Y CONTROLES ============
if st.session_state.get("perdio", False):
    st.error("🚨 El juego terminó porque la inflación salió del rango permitido (±3%). Presiona Recarga la pagina para comenzar de nuevo.", width=600)
    st.image("SIMULACION/sad.jpg", caption="😢 Has perdido el juego y pusiste triste a Julio Velarde", width=600)
    st.stop()

col_datos, col_graficos = st.columns([1, 3])

with col_datos:
    df_j = st.session_state.df_juego
    idx = st.session_state.index

    fecha_actual = df_j.loc[idx, "fecha"] if idx in df_j.index else df_j.loc[idx - 1, "fecha"] + pd.DateOffset(months=1)
    st.markdown(f"### 📆 Periodo actual: **{fecha_actual.strftime('%b-%y')}**")

    # === INPUT de tasa de interés ===
    st.session_state.tasa_input = st.number_input(
        "🔢 Coloca tu tasa de interés nominal (%)", 
        value=st.session_state.tasa_input,
        step=0.25, 
        format="%.2f"
    )

    # === Botones de control ===
    col_btn1, col_btn2 = st.columns(2)
    aceptar = col_btn1.button("✅ ACEPTAR", key="aceptar_btn")
    reiniciar = col_btn2.button("🔁 REINICIAR", key="reiniciar_btn")

    if reiniciar:
        st.session_state.df_juego = df.copy()
        st.session_state.index = START_INDEX
        st.session_state.tasa_input = R + PI_META
        st.session_state.perdio = False
        st.rerun()

    if aceptar:
        # Recuperar parámetros
        a1, a2, a3, a4 = a_params
        b1, b2, b3, b4 = b_params
        c1, c2, c3, c4, c5 = c_params

        # Variables del periodo anterior
        y_t_1 = df_j.loc[idx - 1, "brecha"]
        pi_t_1 = df_j.loc[idx - 1, "infla"]
        pi_t_2 = df_j.loc[idx - 2, "infla"]
        tasa_t_1 = df_j.loc[idx - 1, "tasa"]
        tasa_t_2 = df_j.loc[idx - 2, "tasa"]
        einfla_t_1 = df_j.loc[idx - 1, "einfla"]

        # CALCULAR EXPECTATIVAS SEGÚN LAS ECUACIONES PROPUESTAS
        # Expectativa de inflación: i_real_t = i_nominal_t - E[inflacion_t+1]
        # Despejamos E[inflacion_t+1] = i_nominal_t - i_real_t
        # Como aproximación, usamos la tasa real del periodo anterior
        i_real_previo = tasa_t_1 - einfla_t_1
        einfla_t = st.session_state.tasa_input - i_real_previo
        
        # Expectativa de brecha: E[pbi_t+1] = brecha_t + 0.05 + e
        epbi_t = y_t_1 + 0.05 + np.random.uniform(*ERROR_RANGE)

        # Variables para las ecuaciones
        delta_einfla = einfla_t - df_j.loc[idx - 12, "einfla"]
        gap_real = st.session_state.tasa_input - einfla_t - R
        gap_einfla = einfla_t - PI_META

        # Errores aleatorios
        error_pi = np.random.uniform(*ERROR_RANGE)
        error_y = np.random.uniform(*ERROR_RANGE)
        error_i = np.random.uniform(*ERROR_RANGE)

        # Ecuaciones del modelo
        pi_new = a1 * pi_t_1 + a2 * pi_t_2 + a3 * einfla_t + a4 * y_t_1 + error_pi
        if not (PI_META - 6 <= pi_new <= PI_META ):
            st.session_state.perdio = True
            st.rerun()

        y_new = b1 * y_t_1 + b2 * epbi_t + b3 * gap_real + b4 * delta_einfla + error_y
        tasa_new = c1 * tasa_t_1 + c2 * tasa_t_2 + c3 * (R + PI_META) + c4 * gap_einfla + c5 * y_t_1 + error_i

        # Actualizar DataFrame
        nuevo = df_j.iloc[idx].copy() if idx in df_j.index else df_j.iloc[-1].copy()
        nuevo["infla"] = pi_new
        nuevo["brecha"] = y_new
        nuevo["tasa"] = st.session_state.tasa_input  # Usamos el valor ingresado por el usuario
        nuevo["einfla"] = einfla_t
        nuevo["epbi"] = epbi_t
        nuevo["fecha"] = fecha_actual

        st.session_state.df_juego.loc[idx] = nuevo
        st.session_state.index += 1
        st.rerun()

    # === Mostrar datos del periodo anterior ===
    if idx > 1:
        infla_t = df_j.loc[idx - 1, "infla"]
        brecha_t = df_j.loc[idx - 1, "brecha"]
        einfla_t = df_j.loc[idx - 1, "einfla"]
        epbi_t = df_j.loc[idx - 1, "epbi"]
        tasa_t = df_j.loc[idx - 1, "tasa"]

        st.markdown("### 📊 Datos del periodo anterior")
        st.write(f"πₜ: **{infla_t:.2f}%**")
        st.write(f"Y'ₜ: **{brecha_t:.2f}%**")
        st.write(f"E[πₜ₊₁]: **{einfla_t:.2f}%**")
        st.write(f" E[Yₜ₊₁]: **{epbi_t:.2f}%**")
        st.write(f" iₜ: **{tasa_t:.2f}%**")

# ============ 7. VISUALIZACIÓN ============
with col_graficos:
    st.markdown("## 📈 Visualización del Juego")
    modo = st.radio("Selecciona una vista:", ["Gráficos", "Datos tipo Excel"])

    idx = st.session_state.index
    df_plot = st.session_state.df_juego.copy()
    df_plot = df_plot.iloc[max(0, idx -36):idx + 1]
    df_plot["fecha_fmt"] = df_plot["fecha"].dt.strftime("%b-%y")
    df_plot["tasa_real"] = df_plot["tasa"] - df_plot["einfla"]

    if modo == "Gráficos":
        # 1. Inflación
        fig_infla = go.Figure()
        fig_infla.add_trace(go.Scatter(x=df_plot["fecha_fmt"], y=df_plot["infla"],
                                   name="Inflación Observada (πₜ)", mode="lines+markers"))
        fig_infla.add_trace(go.Scatter(x=df_plot["fecha_fmt"], y=[PI_META]*len(df_plot),
                                   name="Meta Inflación (π*)", line=dict(dash='dash')))
        fig_infla.add_trace(go.Scatter(x=df_plot["fecha_fmt"], y=[PI_META -6]*len(df_plot),
                                   name="Meta Inflación (π*)", line=dict(dash='dash')))
        fig_infla.add_trace(go.Scatter(x=df_plot["fecha_fmt"], y=[0]*len(df_plot),
                               name="Línea base (0%)", line=dict(color="gray", dash="dot")))
        fig_infla.update_layout(title="Inflación Observada (πₜ)", xaxis_title="Fecha", yaxis_title="Inflación (%)")
        st.plotly_chart(fig_infla, use_container_width=True)

        # 2. Brecha del producto
        fig_brecha = go.Figure()
        fig_brecha.add_trace(go.Scatter(x=df_plot["fecha_fmt"], y=df_plot["brecha"],
                                    name="Brecha del Producto (Y'ₜ)", mode="lines+markers"))
        fig_infla.add_trace(go.Scatter(x=df_plot["fecha_fmt"], y=[0]*len(df_plot),
                               name="Línea base (0%)", line=dict(color="red", dash="dot")))
        fig_brecha.update_layout(title="Brecha del Producto (Y'ₜ)", xaxis_title="Fecha", yaxis_title="%")
        st.plotly_chart(fig_brecha, use_container_width=True)

        # 3. Tasa nominal vs real
        fig_tasas = go.Figure()
        fig_tasas.add_trace(go.Scatter(x=df_plot["fecha_fmt"], y=df_plot["tasa"],
                                  name="Tasa Nominal (iₜ)", mode="lines+markers"))
        fig_tasas.add_trace(go.Scatter(x=df_plot["fecha_fmt"], y=df_plot["tasa_real"],
                                  name="Tasa Real (iₜ - E[πₜ₊₁])", mode="lines+markers"))
        fig_tasas.update_layout(title="Tasas de Interés Nominal y Real", xaxis_title="Fecha", yaxis_title="%")
        st.plotly_chart(fig_tasas, use_container_width=True)

    else:
        df_excel = df_plot.copy()
        df_excel["fecha"] = df_excel["fecha"].dt.strftime("%b-%y")
        columnas = ["fecha", "infla", "einfla", "epbi", "brecha", "tasa", "tasa_real"]
        st.dataframe(df_excel[columnas].set_index("fecha"), height=500)