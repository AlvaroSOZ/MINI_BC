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
@st.cache_data
def cargar_datos():
    df = pd.read_csv("SIMULACION/DATOS.csv")

    # Asignar rango de fechas mensual desde septiembre 2003
    df["fecha"] = pd.date_range(start="2003-09-01", periods=len(df), freq="MS")

    # Calcular la brecha de producto con filtro de Hodrick-Prescott
    ciclo, tendencia = hpfilter(df["pbi"], lamb=14400)  # lambda mensual
    df["brecha"] = (df["pbi"] - tendencia) / tendencia * 100

    return df

df_original = cargar_datos()
df = df_original.copy()

# ============ 3. CONSTANTES DEL MODELO ============
PI_META = 4.0      # Meta de inflación
R = 3.5            # Tasa de interés neutral o libre de riesgo
START_INDEX = 256  # Enero 2025 (desde sep-2003 son 256 meses)
ERROR_RANGE = (-0.05, 0.05)  # Rango de errores aleatorios para simulación

# ============ 4. ESTIMACIÓN DE PARÁMETROS ECONOMÉTRICOS ============

@st.cache_data  # Cacheamos esta función para que no se recalculen las variables cada vez que se recargue la app
def preparar_variables(df):
    # Creamos las variables necesarias para las regresiones econométricas

    df["infla_t_1"] = df["infla"].shift(1)  # Inflación rezagada 1 periodo
    df["infla_t_2"] = df["infla"].shift(2)  # Inflación rezagada 2 periodos
    df["einfla_t_1"] = df["einfla"].shift(1)  # Expectativa de inflación rezagada 1 periodo
    df["brecha_t_1"] = df["brecha"].shift(1)  # Brecha del producto rezagada 1 periodo

    df["tasa_t_1"] = df["tasa"].shift(1)  # Tasa de interés rezagada 1 periodo
    df["tasa_t_2"] = df["tasa"].shift(2)  # Tasa de interés rezagada 2 periodos

    df["gap_real"] = df["tasa_t_1"] - df["einfla_t_1"] - R  # Gap real: tasa nominal menos expectativa menos tasa natural

    df["gap_einfla"] = df["einfla_t_1"] - PI_META  # Diferencia entre expectativa y meta inflacionaria

    df["constante"] = R + PI_META  # Valor constante usado en la ecuación de tasa de interés

    df["delta_einfla"] = df["einfla"] - df["einfla"].shift(12)  # Cambio anual de expectativas de inflación

    return df.dropna()  # Eliminamos filas con valores NaN por los rezagos

# Aplicamos el procesamiento a la base de datos
df = preparar_variables(df)

@st.cache_data  # Cacheamos también esta función para eficiencia
def estimar_coeficientes(df):
    # ======== REGRESIÓN 1: Ecuación de inflación ========
    # π_t = a1 π_t−1 + a2 π_t−2 + a3 E[π_t+1] + a4 Y'_t−1 + e

    X_pi = df[["infla_t_1", "infla_t_2", "einfla_t_1", "brecha_t_1"]]  # Variables explicativas
    X_pi = sm.add_constant(X_pi)  # Agregamos constante
    y_pi = df["infla"]  # Variable dependiente: inflación observada
    modelo_pi = sm.OLS(y_pi, X_pi).fit()  # Ajustamos el modelo OLS

    a1 = modelo_pi.params.get("infla_t_1", 0)  # Coeficiente a1: π_t-1
    a2 = modelo_pi.params.get("infla_t_2", 0)  # Coeficiente a2: π_t-2
    a3 = modelo_pi.params.get("einfla_t_1", 0)  # Coeficiente a3: E[π_t+1]
    a4 = modelo_pi.params.get("brecha_t_1", 0)  # Coeficiente a4: Y’_t-1

    # ======== REGRESIÓN 2: Ecuación de brecha del producto ========
    # Y'_t = b1 Y'_t−1 + b2 E[Y_{t+1}] + b3 (i_t - E[π_t+1]) + b4 ΔE[π_t+1] + e

    X_y = df[["brecha_t_1", "epbi", "gap_real", "delta_einfla"]]  # Variables explicativas
    X_y = sm.add_constant(X_y)  # Agregamos constante
    y_y = df["brecha"]  # Variable dependiente: brecha del producto
    modelo_y = sm.OLS(y_y, X_y).fit()  # Ajustamos el modelo OLS

    b1 = modelo_y.params.get("brecha_t_1", 0)  # Coeficiente b1: Y’_t-1
    b2 = modelo_y.params.get("epbi", 0)  # Coeficiente b2: E[Y_{t+1}]
    b3 = modelo_y.params.get("gap_real", 0)  # Coeficiente b3: tasa real ex-ante
    b4 = modelo_y.params.get("delta_einfla", 0)  # Coeficiente b4: cambio en expectativas

    # ======== REGRESIÓN 3: Ecuación de tasa de interés ========
    # i_t = c1 i_t−1 + c2 i_t−2 + c3(r + π*) + c4 (E[π_t+1] - π*) + c5 Y'_t-1 + e

    X_i = df[["tasa_t_1", "tasa_t_2", "constante", "gap_einfla", "brecha_t_1"]]  # Variables explicativas
    X_i = sm.add_constant(X_i)  # Agregamos constante
    y_i = df["tasa"]  # Variable dependiente: tasa de interés nominal
    modelo_i = sm.OLS(y_i, X_i).fit()  # Ajustamos el modelo OLS

    c1 = modelo_i.params.get("tasa_t_1", 0)  # Coeficiente c1: i_t-1
    c2 = modelo_i.params.get("tasa_t_2", 0)  # Coeficiente c2: i_t-2
    c3 = modelo_i.params.get("constante", 0)  # Coeficiente c3: r + π*
    c4 = modelo_i.params.get("gap_einfla", 0)  # Coeficiente c4: brecha de inflación esperada
    c5 = modelo_i.params.get("brecha_t_1", 0)  # Coeficiente c5: brecha del producto

    return (a1, a2, a3, a4), (b1, b2, b3, b4), (c1, c2, c3, c4, c5)  # Devolvemos tupla con los tres conjuntos de coeficientes

# Ejecutamos la estimación y guardamos los parámetros estimados
a_params, b_params, c_params = estimar_coeficientes(df)
##st.write("Parámetros estimados:", {"a_params (π)": a_params, "b_params (Y')": b_params, "c_params (i)": c_params})

# ============ SESIÓN ============
# Inicializar el índice del período de juego (256 ≈ enero 2025)
if "index" not in st.session_state:
    st.session_state.index = START_INDEX  # comienza en 2025-01

# Inicializar el DataFrame con los datos de simulación
if "df_juego" not in st.session_state:
    st.session_state.df_juego = df.copy()

# Buffers para expectativas que se usan y actualizan en cada ronda
if "einfla_buffer" not in st.session_state:
    st.session_state.einfla_buffer = {}  # expectativa de inflación

if "epbi_buffer" not in st.session_state:
    st.session_state.epbi_buffer = {}    # expectativa de brecha producto



###############6
# ============ LAYOUT DE INTERFAZ ============
# Dividir la pantalla: izquierda para datos y controles, derecha para gráficos o tabla
col_datos, col_graficos = st.columns([1, 3])

# ------------ COLUMNA DE DATOS Y CONTROLES ------------
with col_datos:
    df_j = st.session_state.df_juego      # DataFrame con simulación activa
    idx = st.session_state.index          # Índice actual del juego

    # Calcular fecha actual del juego
    fecha_actual = df_j.loc[idx, "fecha"] if idx in df_j.index else df_j.loc[idx - 1, "fecha"] + pd.DateOffset(months=1)

    st.markdown(f"### 📆 Periodo actual: **{fecha_actual.strftime('%b-%y')}**")

    # Input de tasa de interés nominal
    tasa_input = st.number_input("🔢 Coloca tu tasa de interés nominal (%)", step=0.25, format="%.2f")

    # Botones de control
    col_btn1, col_btn2 = st.columns(2)
    if col_btn1.button("✅ ACEPTAR", key="btn_aceptar_input"):
        st.session_state.tasa_input = tasa_input  # Guardar input para procesar después

    if col_btn2.button("🔁 REINICIAR", key="btn_reiniciar_general"):
        st.session_state.df_juego = df.copy()
        st.session_state.index = START_INDEX
        st.session_state.einfla_buffer = {}
        st.session_state.epbi_buffer = {}
        st.rerun()

    # Mostrar variables actuales (tasa, inflación, expectativas, etc.)
    if idx > 1:
        infla_t = df_j.loc[idx - 1, "infla"]
        brecha_t = df_j.loc[idx - 1, "brecha"]
        einfla_t = st.session_state.einfla_buffer.get(idx, np.nan)
        epbi_t = st.session_state.epbi_buffer.get(idx, np.nan)
        tasa_t = df_j.loc[idx - 1, "tasa"]

        st.markdown("### 📊 Datos del periodo anterior")
        st.write(f"Inflación observada πₜ: **{infla_t:.2f}%**")
        st.write(f"Brecha del producto Y'ₜ: **{brecha_t:.2f}%**")
        st.write(f"Expectativa de inflación E[πₜ₊₁]: **{einfla_t:.2f}%**")
        st.write(f"Expectativa de brecha E[Yₜ₊₁]: **{epbi_t:.2f}%**")
        st.write(f"Tasa de interés nominal iₜ: **{tasa_t:.2f}%**")





#####################################################################################################################################
##################################          7   ##############################


col_btn1, col_btn2 = st.columns(2)

if col_btn1.button("✅ ACEPTAR"):
    st.session_state.tasa_input = tasa_input

    # --- Recuperamos parámetros ---
    a1, a2, a3, a4 = a_params
    b1, b2, b3, b4 = b_params
    c1, c2, c3, c4, c5 = c_params

    # --- Variables del periodo anterior ---
    einfla_t = st.session_state.einfla_buffer[idx]
    epbi_t = st.session_state.epbi_buffer[idx]
    y_t_1 = df_j.loc[idx - 1, "brecha"]
    pi_t_1 = df_j.loc[idx - 1, "infla"]
    pi_t_2 = df_j.loc[idx - 2, "infla"]
    tasa_t_1 = df_j.loc[idx - 1, "tasa"]
    tasa_t_2 = df_j.loc[idx - 2, "tasa"]

    # --- Auxiliares ---
    delta_einfla = einfla_t - df_j.loc[idx - 12, "einfla"]
    gap_real = tasa_input - einfla_t - R
    gap_einfla = einfla_t - PI_META

    # --- Errores aleatorios ---
    error_pi = np.random.uniform(*ERROR_RANGE)
    error_y = np.random.uniform(*ERROR_RANGE)
    error_i = np.random.uniform(*ERROR_RANGE)

    # --- Nuevas variables ---
    pi_new = a1 * pi_t_1 + a2 * pi_t_2 + a3 * einfla_t + a4 * y_t_1 + error_pi
    if not (-4 <= pi_new <= 4):
        st.session_state.perdio = True
        st.error("🚨 ¡Has perdido! La inflación salió del rango permitido (-4%, +4%). Reinicia para volver a intentar.")
        st.stop()
    y_new = b1 * y_t_1 + b2 * epbi_t + b3 * gap_real + b4 * delta_einfla + error_y
    tasa_new = c1 * tasa_t_1 + c2 * tasa_t_2 + c3 * (R + PI_META) + c4 * gap_einfla + c5 * y_t_1 + error_i

    # --- Guardar en df_juego ---
    nuevo = df_j.iloc[idx].copy() if idx in df_j.index else df_j.iloc[-1].copy()
    nuevo["infla"] = pi_new
    nuevo["brecha"] = y_new
    nuevo["tasa"] = tasa_new
    nuevo["einfla"] = einfla_t
    nuevo["epbi"] = epbi_t
    nuevo["fecha"] = fecha_actual

    st.session_state.df_juego.loc[idx] = nuevo
    st.session_state.index += 1
    st.rerun()

if col_btn2.button("🔁 REINICIAR"):
    st.session_state.df_juego = df.copy()
    st.session_state.index = START_INDEX
    st.session_state.einfla_buffer = {}
    st.session_state.epbi_buffer = {}
    st.session_state.perdio = False
    st.rerun()

#####################################################################################################################################
##################################        8888888888888   ##############################


with col_graficos:
    st.markdown("## 📈 Visualización")
    modo = st.radio("Selecciona una vista", ["Gráficos", "Datos tipo Excel"])

    df_plot = st.session_state.df_juego.copy()
    df_plot = df_plot.iloc[max(0, idx - 40):idx + 1]
    df_plot["fecha_fmt"] = df_plot["fecha"].dt.strftime("%b-%y")
    df_plot["tasa_real"] = df_plot["tasa"] - df_plot["einfla"]

    if modo == "Gráficos":
        import plotly.graph_objects as go

        # 1. Inflación
        fig_infla = go.Figure()
        fig_infla.add_trace(go.Scatter(x=df_plot["fecha_fmt"], y=df_plot["infla"], name="Inflación"))
        fig_infla.add_trace(go.Scatter(x=df_plot["fecha_fmt"], y=[PI_META]*len(df_plot), name="Meta", line=dict(dash='dot')))
        fig_infla.add_trace(go.Scatter(x=df_plot["fecha_fmt"], y=[PI_META-8]*len(df_plot), name="-4%", line=dict(dash='dot')))
        fig_infla.update_layout(title="Inflación (%)", xaxis_title="Fecha", yaxis_title="Inflación")
        st.plotly_chart(fig_infla, use_container_width=True)

        # 2. Brecha del producto
        fig_brecha = go.Figure()
        fig_brecha.add_trace(go.Scatter(x=df_plot["fecha_fmt"], y=df_plot["brecha"], name="Brecha del Producto"))
        fig_brecha.update_layout(title="Brecha del Producto", xaxis_title="Fecha", yaxis_title="%")
        st.plotly_chart(fig_brecha, use_container_width=True)

        # 3. Tasa nominal
        fig_nom = go.Figure()
        fig_nom.add_trace(go.Scatter(x=df_plot["fecha_fmt"], y=df_plot["tasa"], name="Tasa Nominal"))
        fig_nom.update_layout(title="Tasa de Interés Nominal (%)", xaxis_title="Fecha", yaxis_title="%")
        st.plotly_chart(fig_nom, use_container_width=True)

        # 4. Tasa real
        fig_real = go.Figure()
        fig_real.add_trace(go.Scatter(x=df_plot["fecha_fmt"], y=df_plot["tasa_real"], name="Tasa Real"))
        fig_real.update_layout(title="Tasa de Interés Real (%)", xaxis_title="Fecha", yaxis_title="%")
        st.plotly_chart(fig_real, use_container_width=True)

    else:
        df_excel = df_plot.copy()
        df_excel["fecha"] = df_excel["fecha"].dt.strftime("%b-%y")
        columnas = ["fecha", "infla", "einfla", "epbi", "brecha", "tasa", "tasa_real"]
        st.dataframe(df_excel[columnas].set_index("fecha"), height=500)