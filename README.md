# CursoPhytonProyecto

# Dashboard de Movimiento de Proyectil con Fricción

## Descripción
Dashboard interactivo para comparar métodos numéricos (Euler y RK4) en la resolución de EDOs para un proyectil con resistencia al aire.

## Cómo replicar
1. Clona el repo: `git clone <tu-repo>`.
2. Instala dependencias: `pip install -r requirements.txt`.
3. Ejecuta: `streamlit run app_proyectil.py`.
4. Para publicar: Crea cuenta en Streamlit Cloud, conecta a GitHub, deploya.

## Parámetros por defecto
- v0x=50 m/s, v0y=30 m/s, k=0.01, h=0.01, t_final=10s.

## Hallazgos clave
- RK4 reduce error en ~90% vs Euler.
- Fricción acorta trayectoria ~20%.

Enlace deployado: https://cursophytonproyecto-d5curzvwcbqq4rc3ftv79a.streamlit.app/#2-error-acumulado-vs-tiempo.
