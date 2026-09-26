import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from src.api.schemas import HousingInput

app = FastAPI(
    title="California Housing Prediction API",
    description="API modular para predecir el precio de casas en California usando el mejor modelo de MLflow.",
    version="1.0"
)

# Ruta global del modelo en el Model Registry
MODEL_URI = "mlruns/1/5bc41d30a4b04ec2b85f08c6d163df54/artifacts/model"

try:
    print(f"Cargando el pipeline desde el Model Registry ({MODEL_URI})...")
    pipeline = mlflow.sklearn.load_model(MODEL_URI)
    print("Pipeline cargado exitosamente y listo para predecir.")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    pipeline = None

@app.get("/")
def health_check():
    return {
        "status": "online", 
        "model_loaded": pipeline is not None
    }

@app.post("/predict")
def predict_housing_price(data: HousingInput):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="El modelo no está disponible en el servidor.")
    
    try:
        # 1. Convertir la petición de Pydantic a un diccionario y luego a DataFrame de Pandas
        input_dict = data.model_dump()
        df_input = pd.DataFrame([input_dict])
        
        # 2. El pipeline: escala con StandardScaler y predice con XGBoost internamente
        prediction = pipeline.predict(df_input)
        
        # 3. Retornar el precio estimado (el dataset original mide en cientos de miles de $)
        precio_estimado = float(prediction[0]) * 100000
        
        return {
            "prediction_raw": float(prediction[0]),
            "estimated_price_usd": round(precio_estimado, 2)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error interno durante la predicción: {str(e)}")
