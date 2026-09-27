# California Housing Prediction Project 🏠🤖

Este es un proyecto **End-to-End de Machine Learning** estructurado de forma 100% modular. El objetivo es predecir el precio medio de las viviendas en California utilizando un pipeline robusto que abarca desde la ingesta de datos hasta el despliegue y monitoreo en tiempo real.

---

## 🏗️ Arquitectura de la Solución

El proyecto está dividido en módulos independientes encapsulados en contenedores de **Docker**:

1. **Datos:** Dataset de *California Housing* gestionado de forma nativa.
2. **Entrenamiento y Tracking:** Modelos evaluados (*Linear Regression*, *Random Forest* y *XGBoost*) empaquetados en `Pipelines` de Scikit-Learn y registrados de forma universal (`cloudpickle`) en **MLflow**.
3. **Backend (API):** Servicio de alto rendimiento construido con **FastAPI** que consume el pipeline ganador desde el Model Registry local.
4. **Frontend (App):** Interfaz gráfica interactiva y amigable desarrollada en **Streamlit**.
5. **Monitoreo:** Recolección de métricas de rendimiento y latencia con **Prometheus** y visualización en tableros en tiempo real con **Grafana**.

---

## 📂 Estructura del Proyecto

```text
segunda_prueba/
├── pyproject.toml           # Configuración de dependencias con Poetry 2.0+
├── docker-compose.yml       # Orquestación de toda la infraestructura
├── mlflow.db                # Base de datos SQLite para el Model Registry
├── mlruns/                  # Almacenamiento local de artefactos de MLflow
├── monitoring/
│   └── prometheus.yml       # Configuración de raspado (Scraping) de Prometheus
└── src/
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   └── ingestion.py     # Descarga y división limpia de datos
    ├── models/
    │   ├── __init__.py
    │   └── train.py         # Bucle de entrenamiento y registro en MLflow
    ├── api/
    │   ├── __init__.py
    │   ├── main.py          # API de FastAPI e instrumentación de métricas
    │   ├── schemas.py       # Validaciones de datos de entrada con Pydantic
    │   └── dockerfile
    └── app/
        ├── __init__.py
        ├── app.py           # Interfaz web de usuario (Sliders y formularios)
        └── dockerfile
```

---

## 🚀 Instrucciones de Ejecución

### Prerrequisitos
* Tener instalado **Docker** y **Docker Compose**.
* Tener instalado **Poetry** (opcional, solo para desarrollo local fuera de los contenedores).

### Paso 1: Inicializar la Base de Datos Local
Antes de encender los contenedores, asegúrate de que el archivo local de la base de datos exista para evitar conflictos de permisos de Docker en sistemas basados en Linux:
```bash
touch mlflow.db
```

### Paso 2: Entrenar los Modelos y Generar el Pipeline (Local)
Para registrar los modelos en MLflow y dejar listo el Pipeline ganador de XGBoost, ejecuta el script de entrenamiento desde la raíz utilizando Poetry:
```bash
poetry run python src/models/train.py
```
*Esto generará el experimento en tu panel local y guardará el artefacto serializado en la carpeta `./mlruns`.*

### Paso 3: Desplegar la Infraestructura Completa
Para encender la API, la aplicación web, Prometheus y Grafana de forma centralizada y en segundo plano, ejecuta:
```bash
docker compose up --build -d
```

---

## 🔗 Puertos e Interfaces Disponibles

Una vez que todos los contenedores estén encendidos (`UP`), puedes acceder a los servicios a través de las siguientes URLs en tu navegador:

*   💻 **Aplicación Web (Streamlit):** [http://localhost:8501](http://localhost:8501) *(Interfaz para cotizar viviendas)*
*   🔌 **Documentación de la API (FastAPI):** [http://localhost:8000/docs](http://localhost:8000/docs) *(Swagger interactivo)*
*   📈 **Métricas Crudas de la API:** [http://localhost:8000/metrics](http://localhost:8000/metrics) *(Endpoint expuesto para Prometheus)*
*   🎯 **Panel de Prometheus:** [http://localhost:9090](http://localhost:9090) *(Verificación del estado del scraping)*
*   📊 **Dashboard de Monitoreo (Grafana):** [http://localhost:3000](http://localhost:3000) *(Usuario/Clave por defecto: `admin` / `admin`)*

---

## 🛠️ Tecnologías Utilizadas

*   **Python 3.12**
*   **Poetry** (Gestión de paquetes)
*   **Scikit-Learn** & **XGBoost** (Machine Learning)
*   **MLflow** (MLOps & Model Registry)
*   **FastAPI** & **Pydantic** (Despliegue de API)
*   **Streamlit** (Dashboard del Usuario)
*   **Docker** & **Docker Compose** (Contenedores)
*   **Prometheus** & **Grafana** (Observabilidad y Monitoreo)

---

## 📊 Guía de Rangos y Valores para Pruebas

Para realizar pruebas en la interfaz de **Streamlit** o a través de los **Docs de FastAPI**, se recomienda utilizar valores que se encuentren dentro de los rangos reales del dataset histórico para garantizar predicciones lógicas del modelo:

| Variable | Descripción | Valor Mínimo | Valor Máximo | Valor Sugerido (Medio) |
| :--- | :--- | :---: | :---: | :---: |
| **MedInc** | Ingreso medio en el bloque (en decenas de miles de \$) | 0.50 | 15.00 | 3.50 |
| **HouseAge** | Edad promedio de las viviendas en el bloque (Años) | 1.00 | 52.00 | 28.00 |
| **AveRooms** | Número promedio de habitaciones por hogar | 0.84 | 141.90 | 5.40 |
| **AveBedrms** | Número promedio de dormitorios por hogar | 0.33 | 34.06 | 1.10 |
| **Population** | Población total dentro del bloque | 3.00 | 35,682.00 | 1,425.00 |
| **AveOccup** | Ocupación promedio por hogar (Miembros) | 0.69 | 1,243.30 | 3.00 |
| **Latitude** | Latitud geográfica del bloque (California) | 32.54 | 41.95 | 35.63 |
| **Longitude** | Longitud geográfica del bloque (California) | -124.35 | -114.31 | -119.57 |

*Nota: La variable objetivo original (**target**) mide el valor medio de la casa en cientos de miles de dólares (ej. `2.0` equivale a `$200,000.00 USD`). Tu API multiplica automáticamente el resultado del modelo por `100,000` para entregarte la cifra final formateada en dólares reales.*