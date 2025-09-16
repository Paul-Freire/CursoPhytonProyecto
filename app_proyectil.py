import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.interpolate import interp1d
import os

# Parámetros globales
G = 9.81  # Aceleración gravitacional (m/s²)
T_FINAL = 10.0  # Tiempo final (s)
H_DEFAULT = 0.01  # Paso por defecto (s)
K_DEFAULT = 0.01  # Coef. fricción por defecto (s/m)
V0X_DEFAULT, V0Y_DEFAULT = 50.0, 30.0  # Velocidades iniciales (m/s)
COLORES = {
    'Analítica': '#1E88E5',  # Azul vibrante
    'Euler': '#D81B60',      # Rosa fuerte
    'RK4': '#43A047',       # Verde oscuro
    'Experimental': '#FFB300'  # Amarillo ámbar
}

# Configuración de página
st.set_page_config(page_title="Trayectorias de Proyectil: Análisis Numérico", layout="wide", page_icon="🚀")

# Layout de logos (izquierda, centro, derecha)
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if os.path.exists("logo_izq.png"):
        st.image("logo_izq.png", width=250, use_container_width=False, caption="")
    elif os.path.exists("imagen1.png"):
        st.image("imagen1.png", width=250, use_container_width=False, caption="")
    else:
        st.warning("⚠️ No se encontró 'logo_izq.png' ni 'imagen1.png' para la izquierda.")

with col2:
    if os.path.exists("imagen1.png"):
        st.image("imagen1.png", width=250, use_container_width=False, caption="")
    else:
        st.warning("⚠️ No se encontró 'imagen1.png' en el directorio del proyecto.")

with col3:
    if os.path.exists("logo_der.png"):
        st.image("logo_der.png", width=250, use_container_width=False, caption="")
    elif os.path.exists("imagen1.png"):
        st.image("imagen1.png", width=250, use_container_width=False, caption="")
    else:
        st.warning("⚠️ No se encontró 'logo_der.png' ni 'imagen1.png' para la derecha.")

# Título centrado y llamativo
st.markdown(
    """
    <h1 style="text-align: center;">🚀 Trayectorias de Proyectil: Análisis Numérico 🎯</h1>
    """,
    unsafe_allow_html=True
)
st.divider()

# Función para derivadas (sistema EDO)
@st.cache_data
def f(t, Y, k_fric):
    x, y, vx, vy = Y
    v_mod = np.sqrt(vx**2 + vy**2)
    dxdt = vx
    dydt = vy
    dvxdt = -k_fric * v_mod * vx if v_mod > 0 else 0
    dvydt = -G - k_fric * v_mod * vy if v_mod > 0 else 0
    return np.array([dxdt, dydt, dvxdt, dvydt])

# Método de Euler
@st.cache_data
def euler(Y0, t, h, k_fric):
    n = len(t)
    Y = np.zeros((n, 4))
    Y[0] = Y0
    for i in range(1, n):
        Y[i] = Y[i-1] + h * f(t[i-1], Y[i-1], k_fric)
    return Y

# Método RK4
@st.cache_data
def rk4(Y0, t, h, k_fric):
    n = len(t)
    Y = np.zeros((n, 4))
    Y[0] = Y0
    for i in range(1, n):
        k1 = f(t[i-1], Y[i-1], k_fric)
        k2 = f(t[i-1] + h/2, Y[i-1] + (h/2)*k1, k_fric)
        k3 = f(t[i-1] + h/2, Y[i-1] + (h/2)*k2, k_fric)
        k4 = f(t[i-1] + h, Y[i-1] + h*k3, k_fric)
        Y[i] = Y[i-1] + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
    return Y

# Solución sin fricción (analítica, serializable)
@st.cache_data
def solucion_analitica(v0x, v0y, g=G, t_final=T_FINAL, h=H_DEFAULT):
    t = np.arange(0, t_final + h, h)
    x_ideal = v0x * t
    y_ideal = v0y * t - 0.5 * g * t**2
    vx_ideal = np.full_like(t, v0x)
    vy_ideal = v0y - g * t
    t_vuelo = (2 * v0y) / g
    df_ideal = pd.DataFrame({
        'tiempo': t,
        'x': x_ideal,
        'y': y_ideal,
        'vx': vx_ideal,
        'vy': vy_ideal
    })
    if not os.path.exists('datos_ideal.csv'):
        df_ideal.to_csv('datos_ideal.csv', index=False)
    return df_ideal, t_vuelo

# Sidebar con filtros
st.sidebar.header("Archivo de datos (opcional)")
uploaded = st.sidebar.file_uploader("Cargar CSV experimental (.csv)", type=["csv"])
if uploaded:
    df_exp = pd.read_csv(uploaded)
    df_exp = df_exp.dropna(subset=['tiempo', 'x', 'y'])
    st.sidebar.write(f"Datos experimentales cargados: {len(df_exp)} filas")
else:
    df_exp = None

st.sidebar.header("Filtros")
h = st.sidebar.slider("Paso de tiempo (h, s)", 0.001, 0.1, H_DEFAULT)
if h > 0.1:
    st.warning("⚠️ Paso de tiempo (h) demasiado grande puede causar inestabilidad numérica. Usa h ≤ 0.1.")
    st.stop()
k = st.sidebar.slider("Coef. fricción (k, s/m)", 0.0, 0.05, K_DEFAULT)
v0x = st.sidebar.slider("Vel. inicial x (m/s)", 20.0, 100.0, V0X_DEFAULT)
v0y = st.sidebar.slider("Vel. inicial y (m/s)", 10.0, 50.0, V0Y_DEFAULT)
metodo = st.sidebar.selectbox("Método numérico", ["Euler", "RK4"])
t_vuelo_ideal = (2 * v0y) / G
t_max_limit = min(T_FINAL, t_vuelo_ideal + 1.0)
t_min, t_max = st.sidebar.slider("Rango de tiempo (s)", 0.0, t_max_limit, (0.0, t_max_limit))

# Cálculos principales
t = np.arange(0, T_FINAL + h, h)
Y0 = np.array([0, 0, v0x, v0y])
if metodo == "Euler":
    Y = euler(Y0, t, h, k)
else:
    Y = rk4(Y0, t, h, k)

df_ideal, t_vuelo_ideal = solucion_analitica(v0x, v0y)
mask_t = (t >= t_min) & (t <= t_max)
df_filtrado = pd.DataFrame({
    'tiempo': t[mask_t],
    'x': Y[mask_t, 0],
    'y': Y[mask_t, 1],
    'vx': Y[mask_t, 2],
    'vy': Y[mask_t, 3]
})

mask_ideal_t = (df_ideal['tiempo'] >= t_min) & (df_ideal['tiempo'] <= t_max)
df_ideal_filtrado = df_ideal[mask_ideal_t]
if len(df_filtrado) == 0:
    st.warning("⚠️ Los filtros seleccionados no generan datos. Ajusta el rango de tiempo.")
    st.stop()

# KPIs
col1, col2, col3, col4 = st.columns(4)
dist_recorrida = np.sqrt(Y[mask_t, 0][-1]**2 + Y[mask_t, 1][-1]**2) if len(Y[mask_t]) > 0 else 0
t_vuelo = t[mask_t][np.argmax(Y[mask_t, 1] < 0)] if np.any(Y[mask_t, 1] < 0) else t_max
if len(df_ideal_filtrado) > 0:
    y_interp = interp1d(t[mask_t], Y[mask_t, 1], kind='linear', fill_value='extrapolate')
    error_y = np.abs(y_interp(df_ideal_filtrado['tiempo']) - df_ideal_filtrado['y'])
    error_max = np.max(error_y)
    error_pct = (error_max / np.max(np.abs(df_ideal_filtrado['y']))) * 100 if np.max(np.abs(df_ideal_filtrado['y'])) > 0 else 0
else:
    error_max, error_pct = 0, 0
energia_final = 0.5 * (Y[mask_t, 2][-1]**2 + Y[mask_t, 3][-1]**2) if len(Y[mask_t]) > 0 else 0

with col1:
    st.metric("Distancia Recorrida (m)", f"{dist_recorrida:.2f}")
with col2:
    st.metric("Tiempo de Vuelo Aprox. (s)", f"{t_vuelo:.2f}")
with col3:
    st.metric("Error Máx. en y (%)", f"{error_pct:.2f}")
with col4:
    st.metric("Energía Cinética Final (J)", f"{energia_final:.2f}")
st.divider()

# Gráficos
col_g1, col_g2 = st.columns(2)
with col_g1:
    st.subheader("1. Trayectorias (x vs y)")
    modo_tray = st.radio("Modo de visualización", ["Absoluto", "Normalizado"], horizontal=True, key="tray")
    fig_tray = go.Figure()
    x_ideal, y_ideal = df_ideal_filtrado['x'], df_ideal_filtrado['y']
    x_num, y_num = df_filtrado['x'], df_filtrado['y']
    if modo_tray == "Normalizado":
        x_max = max(x_ideal.max(), x_num.max())
        y_max = max(y_ideal.max(), y_num.max())
        x_ideal, y_ideal = x_ideal/x_max, y_ideal/y_max
        x_num, y_num = x_num/x_max, y_num/y_max
        x_label, y_label = "x/x_max", "y/y_max"
    else:
        x_label, y_label = "x (m)", "y (m)"
    fig_tray.add_trace(go.Scatter(x=x_ideal, y=y_ideal, mode='lines', name='Analítica (sin fricción)', line=dict(dash='dash', color=COLORES['Analítica'])))
    fig_tray.add_trace(go.Scatter(x=x_num, y=y_num, mode='lines', name=f'{metodo} (con fricción)', line=dict(color=COLORES[metodo])))
    if df_exp is not None:
        x_exp, y_exp = df_exp['x'], df_exp['y']
        if modo_tray == "Normalizado":
            x_exp, y_exp = x_exp/x_max, y_exp/y_max
        fig_tray.add_trace(go.Scatter(x=x_exp, y=y_exp, mode='markers', name='Experimental', marker=dict(size=5, color=COLORES['Experimental'])))
    fig_tray.update_layout(
        title="Trayectorias Comparadas",
        xaxis_title=x_label,
        yaxis_title=y_label,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#FFFFFF' if st.get_option('theme.backgroundColor') == '#0E1117' else '#000000'),
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    st.plotly_chart(fig_tray, use_container_width=True)

with col_g2:
    st.subheader("2. Error Acumulado vs. Tiempo")
    modo_error = st.radio("Modo de visualización", ["Absoluto", "Normalizado"], horizontal=True, key="error")
    if len(df_ideal_filtrado) > 0:
        error_t = np.abs(Y[mask_t, 1] - interp1d(df_ideal_filtrado['tiempo'], df_ideal_filtrado['y'], kind='linear', fill_value='extrapolate')(t[mask_t]))
        if modo_error == "Normalizado":
            error_max_val = error_t.max() if error_t.max() > 0 else 1.0
            error_t = error_t / error_max_val
            y_label = "Error / Error_max"
        else:
            y_label = "Error absoluto en y (m)"
        fig_error = px.line(x=t[mask_t], y=error_t, title="Error en y vs Tiempo")
        fig_error.update_layout(
            xaxis_title="Tiempo (s)",
            yaxis_title=y_label,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FFFFFF' if st.get_option('theme.backgroundColor') == '#0E1117' else '#000000'),
            height=400,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        st.plotly_chart(fig_error, use_container_width=True)
    else:
        st.info("No hay datos para el gráfico de error con los filtros seleccionados.")

st.subheader("3. Barras Comparativas: Distancia por Método y Fricción")
Y_euler = euler(Y0, t, h, k)
Y_rk4 = rk4(Y0, t, h, k)
dist_euler = np.sqrt(Y_euler[mask_t, 0][-1]**2 + Y_euler[mask_t, 1][-1]**2) if len(Y_euler[mask_t]) > 0 else 0
dist_rk4 = np.sqrt(Y_rk4[mask_t, 0][-1]**2 + Y_rk4[mask_t, 1][-1]**2) if len(Y_rk4[mask_t]) > 0 else 0
dist_sin_fric = np.sqrt(df_ideal_filtrado['x'].iloc[-1]**2 + df_ideal_filtrado['y'].iloc[-1]**2) if len(df_ideal_filtrado) > 0 else 0
df_barras = pd.DataFrame({
    'Método': ['Euler con fricción', 'RK4 con fricción', 'Analítica sin fricción'],
    'Distancia': [dist_euler, dist_rk4, dist_sin_fric]
})
fig_barras = px.bar(
    df_barras,
    x='Método',
    y='Distancia',
    title="Comparación de Distancias",
    color='Método',
    color_discrete_map=COLORES
)
fig_barras.update_layout(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#FFFFFF' if st.get_option('theme.backgroundColor') == '#0E1117' else '#000000'),
    height=400,
    margin=dict(l=50, r=50, t=50, b=50)
)
st.plotly_chart(fig_barras, use_container_width=True)
st.divider()

# Tabla y descarga
st.subheader("Tabla de Detalles (Datos Filtrados)")
st.dataframe(df_filtrado.head(100), use_container_width=True)
st.caption("Mostrando hasta 100 filas. Descarga el CSV para ver todos los datos.")
csv = df_filtrado.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Descargar CSV Filtrado", csv, "datos_proyectil_filtrados.csv", "text/csv")
st.divider()

# Sección Hallazgos
st.subheader("Hallazgos y Conclusiones")
st.write("""
- **Diferencias entre métodos:** Euler (O(h)) acumula error lineal, menos estable para EDOs con fricción. RK4 (O(h^4)) reduce error ~10-100x, ideal para precisión.
- **Impacto de la fricción:** Reduce distancia recorrida ~15-25% vs. caso analítico, curvando trayectoria hacia abajo. A mayor k, mayor desviación.
- **Recomendaciones:** Usar RK4 para simulaciones críticas (e.g., balística). Reducir h mejora precisión, pero aumenta tiempo de cómputo.
- **Vs. Analítica:** Error crece con tiempo por fricción y método numérico. Sin fricción, la solución es exacta (parábola).
""")

# Sección Cómo Replicar
with st.expander("Cómo Replicar y Publicar"):
    st.markdown("""
    1. Instala: `pip install -r requirements.txt`.
    2. Asegúrate de tener `imagen1.png` (y opcionalmente `logo_izq.png`, `logo_der.png`) en el directorio.
    3. Ejecuta local: `streamlit run app_proyectil.py`.
    4. Publica: Sube a GitHub, ve a [share.streamlit.io](https://share.streamlit.io), conecta repo y deploya.
    5. Enlace ejemplo: [Tu-app.streamlit.app].
    """)
