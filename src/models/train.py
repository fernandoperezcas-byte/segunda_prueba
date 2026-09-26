import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from src.data.ingestion import load_raw_data, split_data

def run_training():
    # 1. Configuración del Experimento en MLflow
    mlflow.set_experiment("California_Housing_Project")
    
    # 2. Flujo de datos utilizando los módulos de 'src/data'
    print("Cargando y procesando datos...")
    df = load_raw_data()
    X_train, X_test, y_train, y_test = split_data(df)
        
    # 3. Definición de los Pipelines (Escalador + Modelo)
    pipelines = {
        "Linear_Regression": Pipeline([
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ]),
        "Random_Forest": Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42))
        ]),
        "XGBoost": Pipeline([
            ('scaler', StandardScaler()),
            ('model', XGBRegressor(n_estimators=100, max_depth=10, learning_rate=0.1, random_state=42))
        ])
    }
    
    best_r2 = -float("inf")
    best_model_name = None
    
    # 4. Bucle de entrenamiento y registro en MLflow
    for model_name, pipeline in pipelines.items():
        # Iniciamos un run individual para cada modelo dentro del experimento
        with mlflow.start_run(run_name=model_name):
            print(f"Entrenando {model_name}...")
            
            pipeline.fit(X_train, y_train) # Entrenar
            
            predictions = pipeline.predict(X_test) # Predecir
            
            # Evaluar
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            print(f"{model_name} -> MAE: {mae:.4f}, R2: {r2:.4f}")
            
            # Registrar hiperparámetros específicos
            model_step = pipeline.named_steps['model']
            if hasattr(model_step, "n_estimators"):
                mlflow.log_param("n_estimators", model_step.get_params()["n_estimators"])
            if hasattr(model_step, "max_depth"):
                mlflow.log_param("max_depth", model_step.get_params()["max_depth"])
                
            # Registrar métricas en MLflow
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            
            # Registrar el artefacto del modelo
            mlflow.sklearn.log_model(pipeline, name="model", serialization_format="cloudpickle")

            # Identificar el mejor modelo basado en R²
            if r2 > best_r2:
                best_r2 = r2
                best_model_name = model_name

    print(f"\nEl mejor modelo es: {best_model_name} con un R2 de {best_r2:.4f}")

if __name__ == "__main__":
    run_training()
