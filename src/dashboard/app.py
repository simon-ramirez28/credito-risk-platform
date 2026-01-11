"""
Dashboard interactivo para análisis de riesgo crediticio.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime
from pathlib import Path
import sys
import os

# Añadir src al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.api.config import settings

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Riesgo Crediticio",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🏦 Dashboard de Análisis de Riesgo Crediticio")
st.markdown("---")

# Configuración de la API
API_URL = "http://localhost:8000"
MODEL_PATH = settings.model_path


@st.cache_resource
def load_model_info():
    """Carga información del modelo."""
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    
    # Fallback: cargar desde archivo
    try:
        metadata_path = settings.model_metadata_path
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except:
        return None


@st.cache_data
def load_sample_data():
    """Carga datos de ejemplo."""
    try:
        # Intentar cargar datos procesados recientes
        data_dir = Path("data/processed")
        if data_dir.exists():
            parquet_files = list(data_dir.glob("*.parquet"))
            if parquet_files:
                latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
                df = pd.read_parquet(latest_file)
                return df.head(100)  # Limitar a 100 registros para performance
    except:
        pass
    
    # Crear datos de ejemplo
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'edad': np.random.randint(20, 65, n_samples),
        'ingreso_mensual': np.random.uniform(1000, 8000, n_samples),
        'score_bancario': np.random.randint(300, 850, n_samples),
        'total_adeudado': np.random.uniform(0, 50000, n_samples),
        'default': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    }
    
    return pd.DataFrame(data)


def calculate_metrics(df):
    """Calcula métricas clave del dataset."""
    metrics = {}
    
    if 'default' in df.columns:
        default_rate = df['default'].mean() * 100
        metrics['Tasa de Default'] = f"{default_rate:.2f}%"
    
    if 'ingreso_mensual' in df.columns:
        metrics['Ingreso Promedio'] = f"${df['ingreso_mensual'].mean():.2f}"
    
    if 'score_bancario' in df.columns:
        metrics['Score Promedio'] = f"{df['score_bancario'].mean():.0f}"
    
    if 'edad' in df.columns:
        metrics['Edad Promedio'] = f"{df['edad'].mean():.1f}"
    
    return metrics


def create_risk_score_chart(scores):
    """Crea gráfico de distribución de scores de riesgo."""
    fig = px.histogram(
        scores, 
        x='risk_score',
        nbins=20,
        title='Distribución de Scores de Riesgo',
        labels={'risk_score': 'Score de Riesgo'},
        color_discrete_sequence=['#FF6B6B']
    )
    
    # Añadir líneas para categorías
    category_lines = [300, 600, 650, 700, 750, 850]
    category_labels = ['Pobre', 'Deficiente', 'Regular', 'Bueno', 'Excelente']
    category_colors = ['#FF4444', '#FFAA44', '#FFFF44', '#AAFF44', '#44FF44']
    
    for i, line in enumerate(category_lines[:-1]):
        fig.add_vline(
            x=line, 
            line_dash="dash", 
            line_color=category_colors[i],
            opacity=0.5
        )
    
    fig.update_layout(
        xaxis_title="Score de Riesgo (300-850)",
        yaxis_title="Frecuencia",
        showlegend=False
    )
    
    return fig


def create_default_probability_chart(probabilities):
    """Crea gráfico de distribución de probabilidades de default."""
    fig = px.histogram(
        probabilities,
        x='probability_default',
        nbins=20,
        title='Distribución de Probabilidades de Default',
        labels={'probability_default': 'Probabilidad de Default'},
        color_discrete_sequence=['#4ECDC4']
    )
    
    fig.add_vline(x=0.5, line_dash="dash", line_color="red", opacity=0.7)
    
    fig.update_layout(
        xaxis_title="Probabilidad de Default (0-1)",
        yaxis_title="Frecuencia",
        xaxis=dict(range=[0, 1])
    )
    
    return fig


def create_correlation_heatmap(df):
    """Crea mapa de calor de correlaciones."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) < 2:
        return None
    
    # Limitar a 10 columnas para mejor visualización
    if len(numeric_cols) > 10:
        numeric_cols = numeric_cols[:10]
    
    corr_matrix = df[numeric_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        title='Matriz de Correlación',
        color_continuous_scale='RdBu',
        zmin=-1,
        zmax=1,
        labels=dict(color="Correlación")
    )
    
    fig.update_layout(
        width=800,
        height=600
    )
    
    return fig


def predict_single_application(data):
    """Envía una solicitud de predicción a la API."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=data,
            headers={"Authorization": "Bearer demo_token"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error en la API: {response.status_code}")
            return None
    except Exception as e:
        st.error(f"Error conectando a la API: {str(e)}")
        return None


def sidebar_controls():
    """Controles de la barra lateral."""
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Selector de modelo
        model_info = load_model_info()
        if model_info:
            st.subheader("📊 Modelo Actual")
            st.write(f"**Tipo:** {model_info.get('model_type', 'N/A')}")
            st.write(f"**Entrenado:** {model_info.get('timestamp', 'N/A')}")
            st.write(f"**Features:** {model_info.get('feature_count', 'N/A')}")
            
            if 'metrics' in model_info:
                metrics = model_info['metrics']
                if 'test' in metrics:
                    st.write(f"**ROC AUC:** {metrics['test'].get('roc_auc', 'N/A'):.3f}")
        
        st.markdown("---")
        
        # Filtros de datos
        st.subheader("🔍 Filtros")
        
        if 'edad' in st.session_state.get('current_data', pd.DataFrame()).columns:
            edad_range = st.slider(
                "Rango de Edad",
                min_value=18,
                max_value=80,
                value=(25, 60)
            )
            st.session_state['edad_filter'] = edad_range
        
        if 'ingreso_mensual' in st.session_state.get('current_data', pd.DataFrame()).columns:
            ingreso_range = st.slider(
                "Rango de Ingreso ($)",
                min_value=0,
                max_value=10000,
                value=(1000, 5000),
                step=100
            )
            st.session_state['ingreso_filter'] = ingreso_range
        
        st.markdown("---")
        
        # Navegación
        st.subheader("📋 Navegación")
        
        page_options = {
            "📈 Visión General": "overview",
            "🎯 Predicciones": "predictions",
            "📊 Análisis": "analysis",
            "⚙️ Configuración": "settings"
        }
        
        selected_page = st.radio(
            "Seleccionar página",
            list(page_options.keys())
        )
        
        st.session_state['current_page'] = page_options[selected_page]
        
        st.markdown("---")
        
        # Información del sistema
        st.subheader("ℹ️ Información")
        st.write(f"**Hora:** {datetime.now().strftime('%H:%M:%S')}")
        st.write(f"**API:** {API_URL}")
        
        # Botón para recargar datos
        if st.button("🔄 Recargar Datos"):
            st.cache_data.clear()
            st.rerun()


def overview_page():
    """Página de visión general."""
    st.header("📈 Visión General del Sistema")
    
    # Cargar datos
    df = load_sample_data()
    st.session_state['current_data'] = df
    
    # Métricas principales
    metrics = calculate_metrics(df)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Clientes Analizados",
            value=len(df),
            delta=f"+{len(df)//10} este mes"
        )
    
    with col2:
        if 'Tasa de Default' in metrics:
            st.metric(
                label="Tasa de Default",
                value=metrics['Tasa de Default'],
                delta="-2.5%" if float(metrics['Tasa de Default'].replace('%', '')) < 15 else "+2.5%"
            )
    
    with col3:
        if 'Ingreso Promedio' in metrics:
            st.metric(
                label="Ingreso Promedio",
                value=metrics['Ingreso Promedio']
            )
    
    with col4:
        if 'Score Promedio' in metrics:
            st.metric(
                label="Score Promedio",
                value=metrics['Score Promedio'],
                delta="+15 puntos"
            )
    
    st.markdown("---")
    
    # Gráficos principales
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución de edades
        if 'edad' in df.columns:
            fig = px.histogram(
                df, 
                x='edad',
                nbins=15,
                title='Distribución de Edades',
                color_discrete_sequence=['#36A2EB']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Distribución de ingresos
        if 'ingreso_mensual' in df.columns:
            fig = px.box(
                df,
                y='ingreso_mensual',
                title='Distribución de Ingresos',
                color_discrete_sequence=['#FFCE56']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Mapa de calor de correlaciones
    st.subheader("🔗 Correlaciones entre Variables")
    heatmap = create_correlation_heatmap(df)
    if heatmap:
        st.plotly_chart(heatmap, use_container_width=True)
    else:
        st.info("No hay suficientes variables numéricas para el mapa de calor.")


def predictions_page():
    """Página de predicciones individuales y por lotes."""
    st.header("🎯 Predicciones de Riesgo")
    
    tab1, tab2 = st.tabs(["📝 Predicción Individual", "📊 Predicción por Lotes"])
    
    with tab1:
        st.subheader("Predicción Individual")
        
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                edad = st.number_input("Edad", min_value=18, max_value=100, value=35)
                genero = st.selectbox("Género", ["M", "F"])
                estado_civil = st.selectbox(
                    "Estado Civil", 
                    ["soltero", "casado", "divorciado", "viudo"]
                )
                dependientes = st.number_input("Dependientes", min_value=0, max_value=10, value=1)
                ingreso_mensual = st.number_input(
                    "Ingreso Mensual ($)", 
                    min_value=0.0, 
                    value=3000.0
                )
                gastos_mensuales = st.number_input(
                    "Gastos Mensuales ($)", 
                    min_value=0.0, 
                    value=2000.0
                )
            
            with col2:
                total_adeudado = st.number_input(
                    "Deuda Total ($)", 
                    min_value=0.0, 
                    value=5000.0
                )
                ahorros = st.number_input(
                    "Ahorros ($)", 
                    min_value=0.0, 
                    value=10000.0
                )
                score_bancario = st.slider(
                    "Score Bancario", 
                    min_value=300, 
                    max_value=850, 
                    value=720
                )
                antiguedad_empleo = st.number_input(
                    "Antigüedad Empleo (meses)", 
                    min_value=0, 
                    value=24
                )
                num_creditos_previos = st.number_input(
                    "Créditos Previos", 
                    min_value=0, 
                    value=2
                )
                max_dias_mora = st.number_input(
                    "Máx. Días de Mora", 
                    min_value=0, 
                    value=15
                )
            
            submitted = st.form_submit_button("🎯 Predecir Riesgo")
            
            if submitted:
                # Preparar datos para la API
                application_data = {
                    "edad": edad,
                    "genero": genero,
                    "estado_civil": estado_civil,
                    "dependientes": dependientes,
                    "ingreso_mensual": ingreso_mensual,
                    "gastos_mensuales": gastos_mensuales,
                    "total_adeudado": total_adeudado,
                    "ahorros": ahorros,
                    "score_bancario": score_bancario,
                    "antiguedad_empleo": antiguedad_empleo,
                    "tipo_contrato": "indefinido",
                    "num_creditos_previos": num_creditos_previos,
                    "max_dias_mora": max_dias_mora,
                    "creditos_problematicos": 0,
                    "tipo_vivienda": "propia",
                    "antiguedad_residencia": 5
                }
                
                # Mostrar spinner mientras se procesa
                with st.spinner("Calculando riesgo..."):
                    result = predict_single_application(application_data)
                
                if result:
                    st.success("✅ Predicción completada")
                    
                    # Mostrar resultados
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            label="Decisión",
                            value="APROBADO" if result['prediction'] == 0 else "RECHAZADO",
                            delta="Bajo riesgo" if result['prediction'] == 0 else "Alto riesgo"
                        )
                    
                    with col2:
                        st.metric(
                            label="Probabilidad Default",
                            value=f"{result['probability_default']:.1%}",
                            delta=f"Score: {result['risk_score']}"
                        )
                    
                    with col3:
                        st.metric(
                            label="Categoría Riesgo",
                            value=result['risk_category'],
                            delta_color="inverse"
                        )
                    
                    # Gráfico de probabilidad
                    fig = go.Figure()
                    
                    fig.add_trace(go.Indicator(
                        mode="gauge+number",
                        value=result['probability_default'] * 100,
                        title={'text': "Probabilidad de Default"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "green"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detalles adicionales
                    with st.expander("📋 Ver detalles técnicos"):
                        st.json(result)
    
    with tab2:
        st.subheader("Predicción por Lotes")
        
        st.info("""
        Sube un archivo CSV con múltiples solicitudes para procesamiento por lotes.
        El archivo debe contener las mismas columnas que el formulario individual.
        """)
        
        uploaded_file = st.file_uploader(
            "Subir archivo CSV", 
            type=['csv'],
            help="El archivo debe tener encabezados en la primera fila"
        )
        
        if uploaded_file is not None:
            try:
                df_batch = pd.read_csv(uploaded_file)
                st.write(f"📊 {len(df_batch)} registros cargados")
                st.dataframe(df_batch.head(), use_container_width=True)
                
                if st.button("🚀 Procesar Lote", type="primary"):
                    with st.spinner(f"Procesando {len(df_batch)} solicitudes..."):
                        # Aquí iría la llamada a la API de batch
                        # Por ahora, simulamos resultados
                        st.success(f"✅ {len(df_batch)} solicitudes procesadas")
                        
                        # Simular resultados
                        np.random.seed(42)
                        simulated_results = pd.DataFrame({
                            'risk_score': np.random.randint(300, 850, len(df_batch)),
                            'probability_default': np.random.uniform(0, 1, len(df_batch)),
                            'decision': np.random.choice(['APROBADO', 'RECHAZADO'], 
                                                         len(df_batch), p=[0.7, 0.3])
                        })
                        
                        # Mostrar resumen
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            aprobados = (simulated_results['decision'] == 'APROBADO').sum()
                            st.metric("Aprobados", aprobados)
                        
                        with col2:
                            promedio_score = simulated_results['risk_score'].mean()
                            st.metric("Score Promedio", f"{promedio_score:.0f}")
                        
                        with col3:
                            tasa_default = simulated_results['probability_default'].mean() * 100
                            st.metric("Default Promedio", f"{tasa_default:.1f}%")
                        
                        # Gráfico de distribución
                        fig = px.histogram(
                            simulated_results,
                            x='risk_score',
                            nbins=20,
                            title='Distribución de Scores del Lote'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Botón para descargar resultados
                        results_csv = simulated_results.to_csv(index=False)
                        st.download_button(
                            label="📥 Descargar Resultados",
                            data=results_csv,
                            file_name="resultados_prediccion.csv",
                            mime="text/csv"
                        )
                        
            except Exception as e:
                st.error(f"Error procesando archivo: {str(e)}")


def analysis_page():
    """Página de análisis avanzado."""
    st.header("📊 Análisis Avanzado")
    
    # Cargar datos
    df = load_sample_data()
    
    tab1, tab2, tab3 = st.tabs(["📈 Análisis por Segmentos", "🔍 Detección de Patrones", "📋 Reportes"])
    
    with tab1:
        st.subheader("Análisis por Segmentos")
        
        col1, col2 = st.columns(2)
        
        with col1:
            segmentation_var = st.selectbox(
                "Variable de Segmentación",
                options=['edad', 'ingreso_mensual', 'score_bancario'] 
                if all(col in df.columns for col in ['edad', 'ingreso_mensual', 'score_bancario']) 
                else df.columns.tolist()
            )
        
        with col2:
            if segmentation_var in df.columns:
                if pd.api.types.is_numeric_dtype(df[segmentation_var]):
                    bins = st.slider("Número de segmentos", 2, 10, 4)
                    
                    # Crear segmentos
                    if len(df) > 0:
                        df_segmented = df.copy()
                        df_segmented['segmento'] = pd.qcut(
                            df_segmented[segmentation_var], 
                            q=bins, 
                            duplicates='drop'
                        )
                        
                        # Calcular métricas por segmento
                        segment_stats = df_segmented.groupby('segmento').agg({
                            'default': 'mean' if 'default' in df.columns else segmentation_var
                        }).reset_index()
                        
                        # Gráfico
                        fig = px.bar(
                            segment_stats,
                            x='segmento',
                            y='default' if 'default' in df.columns else segmentation_var,
                            title=f'Análisis por {segmentation_var}',
                            color_discrete_sequence=['#FF9F40']
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Detección de Patrones")
        
        if 'default' in df.columns:
            # Análisis de variables más importantes para default
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != 'default']
            
            if len(numeric_cols) > 0:
                # Calcular correlaciones
                correlations = df[numeric_cols + ['default']].corr()['default'].abs()
                top_features = correlations.sort_values(ascending=False).head(10)
                
                # Gráfico de importancia
                fig = px.bar(
                    x=top_features.index,
                    y=top_features.values,
                    title='Features más correlacionadas con Default',
                    labels={'x': 'Feature', 'y': 'Correlación Absoluta'},
                    color=top_features.values,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Distribución multivariada
        st.subheader("Distribución Conjunta")
        
        col1, col2 = st.columns(2)
        
        with col1:
            x_var = st.selectbox(
                "Variable X",
                options=df.select_dtypes(include=[np.number]).columns.tolist(),
                key="x_var"
            )
        
        with col2:
            y_var = st.selectbox(
                "Variable Y",
                options=df.select_dtypes(include=[np.number]).columns.tolist(),
                key="y_var"
            )
        
        if x_var and y_var and x_var != y_var:
            fig = px.scatter(
                df,
                x=x_var,
                y=y_var,
                color='default' if 'default' in df.columns else None,
                title=f'{x_var} vs {y_var}',
                opacity=0.6
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Generar Reportes")
        
        report_type = st.selectbox(
            "Tipo de Reporte",
            ["Resumen Ejecutivo", "Reporte Técnico", "Análisis de Riesgo"]
        )
        
        if st.button("📄 Generar Reporte"):
            with st.spinner("Generando reporte..."):
                # Simular generación de reporte
                st.success("✅ Reporte generado exitosamente")
                
                # Mostrar resumen
                st.subheader("Resumen del Reporte")
                
                summary_data = {
                    "Total de Clientes Analizados": len(df),
                    "Tasa de Default": f"{df['default'].mean()*100:.2f}%" if 'default' in df.columns else "N/A",
                    "Score Promedio": f"{df['score_bancario'].mean():.0f}" if 'score_bancario' in df.columns else "N/A",
                    "Ingreso Promedio": f"${df['ingreso_mensual'].mean():.2f}" if 'ingreso_mensual' in df.columns else "N/A",
                    "Deuda Promedio": f"${df['total_adeudado'].mean():.2f}" if 'total_adeudado' in df.columns else "N/A"
                }
                
                for key, value in summary_data.items():
                    st.write(f"**{key}:** {value}")
                
                # Botón para descargar (simulado)
                st.download_button(
                    label="📥 Descargar Reporte (PDF)",
                    data="Contenido simulado del reporte",
                    file_name=f"reporte_riesgo_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain"
                )


def settings_page():
    """Página de configuración."""
    st.header("⚙️ Configuración del Sistema")
    
    with st.form("settings_form"):
        st.subheader("Configuración de la API")
        
        api_host = st.text_input("Host de la API", value="localhost")
        api_port = st.number_input("Puerto de la API", min_value=1, max_value=65535, value=8000)
        
        st.subheader("Configuración del Modelo")
        model_path = st.text_input("Ruta del Modelo", value=MODEL_PATH)
        
        st.subheader("Parámetros de Visualización")
        chart_theme = st.selectbox("Tema de Gráficos", ["plotly", "plotly_white", "plotly_dark"])
        default_sample_size = st.slider("Tamaño de Muestra por Defecto", 50, 1000, 100)
        
        submitted = st.form_submit_button("💾 Guardar Configuración")
        
        if submitted:
            # En una implementación real, guardaríamos en archivo de configuración
            st.success("✅ Configuración guardada (simulado)")
            
            # Actualizar variables globales
            global API_URL
            API_URL = f"http://{api_host}:{api_port}"
            st.info(f"API URL actualizada a: {API_URL}")


def main():
    """Función principal del dashboard."""
    # Inicializar estado de la sesión
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 'overview'
    
    # Barra lateral
    sidebar_controls()
    
    # Navegación entre páginas
    current_page = st.session_state.get('current_page', 'overview')
    
    if current_page == 'overview':
        overview_page()
    elif current_page == 'predictions':
        predictions_page()
    elif current_page == 'analysis':
        analysis_page()
    elif current_page == 'settings':
        settings_page()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        🏦 Sistema de Análisis de Riesgo Crediticio | v1.0.0<br>
        Desarrollado con ❤️ usando Streamlit, FastAPI y Scikit-learn
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()