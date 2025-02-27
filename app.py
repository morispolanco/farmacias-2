import streamlit as st
import pandas as pd
import numpy as np
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import io

# Configuración inicial de la página
st.set_page_config(page_title="Inventory Insight - Farmacia Galeno", layout="wide")
st.title("Inventory Insight - Gestión Inteligente de Inventarios")

# Función para cargar y limpiar datos con validación estricta
def load_data(file):
    try:
        df = pd.read_csv(file)
        expected_columns = ['Fecha', 'Producto', 'Ventas', 'Stock', 'Fecha_Vencimiento']
        for col in expected_columns:
            if col not in df.columns:
                st.error(f"El archivo debe contener la columna: {col}")
                return None
        
        # Convertir fechas con manejo estricto
        df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
        df['Fecha_Vencimiento'] = pd.to_datetime(df['Fecha_Vencimiento'], errors='coerce')
        
        # Eliminar filas con fechas inválidas
        df.dropna(subset=['Fecha', 'Ventas', 'Stock', 'Fecha_Vencimiento'], inplace=True)
        
        # Validar que Fecha_Vencimiento sea datetime64
        if not pd.api.types.is_datetime64_any_dtype(df['Fecha_Vencimiento']):
            st.error("La columna 'Fecha_Vencimiento' contiene valores no válidos después de la conversión.")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {e}")
        return None

# Preprocesar datos para rendimiento
def preprocess_data(df):
    return {product: group for product, group in df.groupby('Producto')}

# Función para pronosticar demanda con auto-ARIMA
def forecast_demand(data, product, days=30):
    sales = data[product]['Ventas'].values
    if len(sales) < 30:
        return None, None, f"Se recomiendan al menos 30 días de datos históricos para un pronóstico fiable (disponibles: {len(sales)})."
    try:
        model = auto_arima(sales, seasonal=False, suppress_warnings=True, stepwise=True)
        forecast = model.predict(n_periods=days, return_conf_int=True)
        predictions, conf_int = forecast[0], forecast[1]
        return predictions, conf_int, None
    except ValueError as e:
        return None, None, f"Error en el modelo: datos no estacionarios o insuficientes. Prueba con más datos históricos. ({e})"
    except Exception as e:
        return None, None, f"Error en el modelo de pronóstico: {e}"

# Función para generar recomendaciones de reabastecimiento
def suggest_restock(current_stock, predicted_demand, threshold, buffer=1.2):
    predicted_stock = current_stock - predicted_demand
    if predicted_stock < threshold:
        restock_amount = (predicted_demand * buffer) - current_stock
        return max(restock_amount, 0)
    return 0

# Sidebar para configuraciones
st.sidebar.header("Configuración")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo CSV", type="csv")

# Generar CSV de ejemplo con 30 días históricos hasta ayer (26/02/2025)
today = pd.Timestamp.today()  # Current date: 2025-02-27
dates = pd.date_range(start=today - pd.Timedelta(days=30), end=today - pd.Timedelta(days=1), freq='D')
products = ['Paracetamol', 'Ibuprofeno']
sample_data = pd.DataFrame({
    'Fecha': list(dates) * len(products),
    'Producto': [p for p in products for _ in range(30)],
    'Ventas': [10, 12, 15, 8, 9, 11, 14, 13, 10, 12, 15, 8, 9, 11, 14, 13, 10, 12, 15, 8, 9, 11, 14, 13, 10, 12, 15, 8, 9, 11] + 
              [5, 6, 7, 4, 5, 6, 8, 7, 5, 6, 7, 4, 5, 6, 8, 7, 5, 6, 7, 4, 5, 6, 8, 7, 5, 6, 7, 4, 5, 6],
    'Stock': [100, 90, 78, 63, 55, 46, 35, 21, 11, 1, 91, 76, 68, 59, 48, 34, 24, 14, 2, 87, 79, 70, 59, 45, 35, 25, 13, 5, 96, 87] + 
             [80, 75, 69, 65, 60, 55, 47, 40, 35, 30, 74, 70, 65, 60, 52, 45, 40, 35, 29, 75, 70, 65, 57, 50, 43, 38, 31, 27, 72, 67],
    'Fecha_Vencimiento': ['2025-06-01'] * 30 + ['2025-07-15'] * 30
})
sample_csv = sample_data.to_csv(index=False)

# Guía de uso
with st.sidebar.expander("Guía de Uso"):
    st.write("**Formato del Archivo**: Sube un CSV con: Fecha, Producto, Ventas, Stock, Fecha_Vencimiento. Ejemplo: `2025-01-01, Paracetamol, 10, 50, 2025-06-01`.")
    st.write("**Pasos**: 1) Carga el archivo, 2) Selecciona un producto, 3) Ajusta parámetros, 4) Revisa resultados.")
    st.write("**Funciones**: Pronósticos, recomendaciones de stock y alertas de vencimiento.")
    st.download_button(
        label="Descargar CSV de Prueba",
        data=sample_csv,
        file_name="ejemplo_inventario.csv",
        mime="text/csv"
    )

# Opciones de personalización
forecast_days = st.sidebar.slider("Días de Pronóstico", 7, 90, 30)
stock_threshold = st.sidebar.number_input("Umbral de Stock Mínimo", min_value=0, value=10)
expiration_days = st.sidebar.slider("Días para Alerta de Vencimiento", 7, 90, 30)

if uploaded_file is not None:
    data = load_data(uploaded_file)
    if data is None:
        st.warning("Error al cargar el archivo. Usando datos de ejemplo.")
        data = sample_data
else:
    data = sample_data

if data is not None:
    preprocessed_data = preprocess_data(data)
    st.sidebar.success("Datos cargados correctamente")
    products = list(preprocessed_data.keys())
    selected_product = st.sidebar.selectbox("Selecciona un Producto", products)

    # Filtrar datos del producto
    product_data = preprocessed_data[selected_product]

    # Pronóstico de demanda
    st.subheader(f"Análisis para: {selected_product}")
    st.write("### Pronóstico de Demanda")
    with st.expander("¿Qué significa esto?"):
        st.write("Predice ventas futuras con un modelo estadístico (ARIMA). Línea azul: pasado; rojo punteado: futuro; sombra: intervalo de confianza.")

    forecast, conf_int, error = forecast_demand(preprocessed_data, selected_product, forecast_days)
    if forecast is not None:
        forecast_dates = pd.date_range(start=product_data['Fecha'].max() + pd.Timedelta(days=1), periods=forecast_days, freq='D')
        forecast_df = pd.DataFrame({'Fecha': forecast_dates, 'Pronóstico': forecast, 'Lower_CI': conf_int[:, 0], 'Upper_CI': conf_int[:, 1]})

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(product_data['Fecha'], product_data['Ventas'], label='Ventas Históricas', color='blue')
        ax.plot(forecast_df['Fecha'], forecast_df['Pronóstico'], label='Pronóstico', color='red', linestyle='--')
        ax.fill_between(forecast_df['Fecha'], forecast_df['Lower_CI'], forecast_df['Upper_CI'], color='red', alpha=0.1, label='Intervalo de Confianza')
        ax.legend()
        ax.set_title(f"Pronóstico de Demanda para {selected_product}")
        ax.set_xlabel("Fecha")
        ax.set_ylabel("Ventas")
        st.pyplot(fig)
    else:
        st.warning(error)

    # Gestión de inventario
    st.write("### Gestión de Inventario")
    with st.expander("¿Qué significa esto?"):
        st.write("Muestra stock actual, demanda futura, stock esperado y cuánto reabastecer.")

    current_stock = product_data['Stock'].iloc[-1]
    predicted_demand = forecast.sum() if forecast is not None else 0
    predicted_stock = current_stock - predicted_demand
    restock_amount = suggest_restock(current_stock, predicted_demand, stock_threshold)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stock Actual", int(current_stock))
    col2.metric("Demanda Pronosticada", int(predicted_demand))
    col3.metric("Stock Esperado", int(predicted_stock))
    col4.metric("Recomendación de Reabastecimiento", int(restock_amount))

    if predicted_stock < stock_threshold:
        st.warning(f"¡Alerta! Stock esperado ({int(predicted_stock)}) por debajo del umbral ({stock_threshold}). Reabastece {int(restock_amount)} unidades.")

    # Control de vencimientos
    st.write("### Control de Vencimientos")
    with st.expander("¿Qué significa esto?"):
        st.write(f"Identifica productos que vencerán en {expiration_days} días para priorizar acciones.")

    expiration_data = product_data[['Fecha_Vencimiento', 'Stock']].dropna()
    expiration_threshold = pd.Timestamp.today() + pd.Timedelta(days=expiration_days)

    # Debugging output
    st.write("Debugging expiration data:")
    st.write("Expiration data types:", expiration_data.dtypes)
    st.write("Expiration data sample:", expiration_data.head())
    st.write(f"Expiration threshold: {expiration_threshold} (type: {type(expiration_threshold)})")

    # Robust comparison with type checking
    valid_dates = expiration_data['Fecha_Vencimiento'].notna()
    if not valid_dates.all():
        st.warning("Algunas fechas en 'Fecha_Vencimiento' son inválidas y serán ignoradas.")
    
    expiring_soon = expiration_data[
        valid_dates & 
        (expiration_data['Fecha_Vencimiento'] <= expiration_threshold.replace(tzinfo=None))
    ]

    if not expiring_soon.empty:
        st.write(f"Productos próximos a vencer (en {expiration_days} días):")
        st.dataframe(expiring_soon.style.format({'Fecha_Vencimiento': '{:%Y-%m-%d}'}))
    else:
        st.success(f"No hay productos próximos a vencer en los próximos {expiration_days} días.")

    # Exportar reporte con detalles de pronóstico
    st.write("### Descargar Reporte")
    report_data = {
        'Producto': [selected_product],
        'Stock Actual': [current_stock],
        'Demanda Pronosticada': [predicted_demand],
        'Stock Esperado': [predicted_stock],
        'Reabastecimiento Sugerido': [restock_amount],
        'Productos por Vencer': [len(expiring_soon)]
    }
    report_df = pd.DataFrame(report_data)
    if forecast is not None:
        forecast_details = forecast_df[['Fecha', 'Pronóstico']].rename(columns={'Pronóstico': 'Ventas Pronosticadas'})
        report_df = pd.concat([report_df, forecast_details], axis=1)
    csv = report_df.to_csv(index=False)
    st.download_button(
        label="Descargar Reporte CSV",
        data=csv,
        file_name=f"Reporte_{selected_product}_{today.strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# Pie de página
st.sidebar.markdown("---")
st.sidebar.write("Desarrollado por xAI para Farmacia XYZ - 2025")
