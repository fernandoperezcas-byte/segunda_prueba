import streamlit as st
import requests

# Configuración de la página
st.set_page_config(
    page_title="Cotizador de Casas - California",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 Estimador de Precios de Viviendas")
st.write("Mueve los controles inferiores para estimar el valor de una propiedad en California en tiempo real.")

# 1. Definir la URL de tu API de FastAPI
# (Cuando usemos Docker, esta URL cambiará al nombre del contenedor)
API_URL = "http://api:8000/predict"

st.sidebar.header("📍 Ubicación del Bloque")
latitude = st.sidebar.slider("Latitud", 32.5, 42.5, 34.05, step=0.01)
longitude = st.sidebar.slider("Longitud", -124.5, -114.3, -118.24, step=0.01)

st.header("📋 Características de la Propiedad")

# Dividimos en columnas para que se vea más ordenado
col1, col2 = st.columns(2)

with col1:
    med_inc = st.number_input("Ingreso Medio del Bloque (en miles de \$)", min_value=0.5, max_value=15.0, value=3.5, step=0.1)
    house_age = st.slider("Edad Media de las Casas (Años)", 1, 52, 15)
    ave_rooms = st.number_input("Promedio de Habitaciones por Casa", min_value=1.0, max_value=20.0, value=5.2, step=0.1)

with col2:
    population = st.number_input("Población Total del Bloque", min_value=3.0, max_value=50000.0, value=800.0, step=50.0)
    ave_occup = st.number_input("Ocupantes Promedio por Hogar", min_value=1.0, max_value=10.0, value=3.0, step=0.1)
    ave_bedrms = st.number_input("Promedio de Dormitorios por Casa", min_value=0.5, max_value=10.0, value=1.0, step=0.1)

# Botón para cotizar
if st.button("💰 Calcular Precio Estimado", type="primary"):
    # 2. Empaquetar los datos exactamente como los pide el esquema Pydantic de la API
    payload = {
        "MedInc": med_inc,
        "HouseAge": house_age,
        "AveRooms": ave_rooms,
        "AveBedrms": ave_bedrms,
        "Population": population,
        "AveOccup": ave_occup,
        "Latitude": latitude,
        "Longitude": longitude
    }
    
    try:
        # Enviar petición POST a FastAPI
        with st.spinner("Conectando con el modelo de XGBoost..."):
            response = requests.post(API_URL, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            precio_final = result["estimated_price_usd"]
            
            # Mostrar el resultado con un formato monetario bonito
            st.success(f"###Precio Estimado: \${precio_final:,.2f} USD")
            st.metric(label="Valor del bloque (Escala Raw del Dataset)", value=f"{result['prediction_raw']:.4f}")
        else:
            st.error(f"Error de la API ({response.status_code}): {response.json().get('detail', 'Error desconocido')}")
            
    except requests.exceptions.ConnectionError:
        st.error("No se pudo conectar con la API de FastAPI. Asegúrate de que el servidor (`uvicorn`) esté encendido en el puerto 8000.")
