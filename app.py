import streamlit as st
import pandas as pd
import numpy as np
from pmdarima import auto_arima
import matplotlib.pyplot as plt
import sqlite3
import io
from datetime import datetime
import bcrypt

# Configuración inicial de la página
st.set_page_config(page_title="Inventory Insight - Farmacia Galeno", layout="wide")

# Función para conectar a la base de datos SQLite
def get_db_connection():
    conn = sqlite3.connect('inventory.db')
    conn.row_factory = sqlite3.Row
    return conn

# Función para inicializar la base de datos
def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    
    # Crear tabla de inventario
    c.execute('''
        CREATE TABLE IF NOT EXISTS inventory (
            Fecha TEXT,
            Producto TEXT,
            Ventas INTEGER,
            Stock INTEGER,
            Fecha_Vencimiento TEXT
        )
    ''')
    
    # Crear tabla de usuarios
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT
        )
    ''')
    
    # Insertar usuario por defecto si no existe (admin/admin123)
    default_user = 'admin'
    default_password = 'admin123'
    hashed_pw = bcrypt.hashpw(default_password.encode('utf-8'), bcrypt.gensalt())
    c.execute("SELECT COUNT(*) FROM users WHERE username = ?", (default_user,))
    if c.fetchone()[0] == 0:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (default_user, hashed_pw))
    
    # Insertar datos de ejemplo en inventario si está vacía
    c.execute("SELECT COUNT(*) FROM inventory")
    if c.fetchone()[0] == 0:
        today = pd.Timestamp.today()
        dates = pd.date_range(start=today - pd.Timedelta(days=30), end=today - pd.Timedelta(days=1), freq='D')
        products = ['Paracetamol', 'Ibuprofeno']
        sample_data = []
        for i, date in enumerate(dates):
            for product in products:
                ventas = [10, 12, 15, 8, 9, 11, 14, 13, 10, 12, 15, 8, 9, 11, 14, 13, 10, 12, 15, 8, 9, 11, 14, 13, 10, 12, 15, 8, 9, 11][i] if product == 'Paracetamol' else [5, 6, 7, 4, 5, 6, 8, 7, 5, 6, 7, 4, 5, 6, 8, 7, 5, 6, 7, 4, 5, 6, 8, 7, 5, 6, 7, 4, 5, 6][i]
                stock = [100, 90, 78, 63, 55, 46, 35, 21, 11, 1, 91, 76, 68, 59, 48, 34, 24, 14, 2, 87, 79, 70, 59, 45, 35, 25, 13, 5, 96, 87][i] if product == 'Paracetamol' else [80, 75, 69, 65, 60, 55, 47, 40, 35, 30, 74, 70, 65, 60, 52, 45, 40, 35, 29, 75, 70, 65, 57, 50, 43, 38, 31, 27, 72, 67][i]
                fecha_venc = '2025-06-01' if product == 'Paracetamol' else '2025-07-15'
                sample_data.append((date.strftime('%Y-%m-%d'), product, ventas, stock, fecha_venc))
        
        c.executemany("INSERT INTO inventory (Fecha, Producto, Ventas, Stock, Fecha_Vencimiento) VALUES (?, ?, ?, ?, ?)", sample_data)
    
    conn.commit()
    conn.close()

# Función para verificar login
def check_login(username, password):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result:
        stored_password = result['password']
        return bcrypt.checkpw(password.encode('utf-8'), stored_password)
    return False

# Función para cargar datos desde la base de datos
def load_data_from_db():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM inventory", conn)
    conn.close()
    
    df['Fecha'] = pd.to_datetime(df['Fecha'], errors='coerce')
    df['Fecha_Vencimiento'] = pd.to_datetime(df['Fecha_Vencimiento'], errors='coerce')
    
    original_len = len(df)
    df.dropna(subset=['Fecha', 'Ventas', 'Stock', 'Fecha_Vencimiento'], inplace=True)
    if len(df) < original_len:
        st.warning(f"Se eliminaron {original_len - len(df)} filas con fechas o datos clave inválidos.")
    
    if not pd.api.types.is_datetime64_any_dtype(df['Fecha_Vencimiento']):
        st.error("La columna 'Fecha_Vencimiento' contiene valores no válidos.")
        return None
    
    return df

# Función para añadir nueva venta a la base de datos
def add_sale(fecha, producto, ventas, stock, fecha_vencimiento):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT INTO inventory (Fecha, Producto, Ventas, Stock, Fecha_Vencimiento) VALUES (?, ?, ?, ?, ?)",
              (fecha, producto, ventas, stock, fecha_vencimiento))
    conn.commit()
    conn.close()

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

# Inicializar la base de datos
init_db()

# Gestión de estado de sesión
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Login
if not st.session_state.logged_in:
    st.sidebar.header("Iniciar Sesión")
    username = st.sidebar.text_input("Usuario")
    password = st.sidebar.text_input("Contraseña", type="password")
    if st.sidebar.button("Login"):
        if check_login(username, password):
            st.session_state.logged_in = True
            st.sidebar.success("¡Inicio de sesión exitoso!")
        else:
            st.sidebar.error("Usuario o contraseña incorrectos")
else:
    # Mostrar app solo si está autenticado
    st.sidebar.header("Configuración")
    
    # Formulario para añadir nueva venta
    with st.sidebar.expander("Añadir Nueva Venta"):
        fecha = st.date_input("Fecha", value=datetime.today())
        producto = st.text_input("Producto")
        ventas = st.number_input("Ventas", min_value=0, value=0)
        stock = st.number_input("Stock", min_value=0, value=0)
        fecha_vencimiento = st.date_input("Fecha de Vencimiento", value=datetime.today().replace(year=datetime.today().year + 1))
        if st.button("Guardar Venta"):
            add_sale(fecha.strftime('%Y-%m-%d'), producto, ventas, stock, fecha_vencimiento.strftime('%Y-%m-%d'))
            st.success(f"Venta de {producto} guardada correctamente.")
    
    # Botón para cerrar sesión
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state.logged_in = False
        st.sidebar.success("Sesión cerrada")

    # Guía de uso
    with st.sidebar.expander("Guía de Uso"):
        st.write("**Datos**: La aplicación usa una base de datos SQLite ('inventory.db').")
        st.write("**Pasos**: 1) Selecciona un producto, 2) Ajusta parámetros, 3) Revisa resultados, 4) Añade ventas si es necesario.")
        st.write("**Funciones**: Pronósticos, recomendaciones de stock y alertas de vencimiento.")

    # Opciones de personalización
    forecast_days = st.sidebar.slider("Días de Pronóstico", 7, 90, 30)
    stock_threshold = st.sidebar.number_input("Umbral de Stock Mínimo", min_value=0, value=10)
    expiration_days = st.sidebar.slider("Días para Alerta de Vencimiento", 7, 90, 30)

    # Cargar datos desde la base de datos
    data = load_data_from_db()

    if data is not None:
        preprocessed_data = preprocess_data(data)
        st.sidebar.success("Datos cargados correctamente desde la base de datos")
        products = list(preprocessed_data.keys())
        selected_product = st.sidebar.selectbox("Selecciona un Producto", products)

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

        if expiration_data.empty:
            expiring_soon = pd.DataFrame(columns=['Fecha_Vencimiento', 'Stock'])
        else:
            valid_dates = expiration_data['Fecha_Vencimiento'].notna()
            if not valid_dates.all():
                st.warning(f"Se encontraron fechas inválidas en 'Fecha_Vencimiento'. Serán ignoradas.")
            
            try:
                expiring_soon = expiration_data[
                    valid_dates & 
                    (expiration_data['Fecha_Vencimiento'] <= expiration_threshold.replace(tzinfo=None))
                ]
            except Exception as e:
                st.error(f"Error al filtrar fechas de vencimiento: {e}")
                expiring_soon = pd.DataFrame(columns=['Fecha_Vencimiento', 'Stock'])

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
            file_name=f"Reporte_{selected_product}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

# Pie de página
st.sidebar.markdown("---")
st.sidebar.write("Desarrollado por xAI para Farmacia XYZ - 2025")
