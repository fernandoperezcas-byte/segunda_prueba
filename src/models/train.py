import mlflow
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from src.data.data_ingestion import load_raw_data, split_data
from src.data.data_processing import DataProcessor

def eval_metrics(actual: pd.Series, pred: np.ndarray):
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return mae, r2

def run_training():
    # 1. Configuración del Experimento en MLflow
    mlflow.set_experiment("California_Housing_Project")
    
    # 2. Flujo de datos utilizando los módulos de 'src/data'
    print("⏳ Cargando y procesando datos...")
    df = load_raw_data()
    X_train, X_test, y_train, y_test = split_data(df)
    
    processor = DataProcessor()
    X_train_scaled = processor.fit_transform(X_train)
    X_test_scaled = processor.transform(X_test)
    
    # 3. Definición de los 3 modelos a evaluar con sus hiperparámetros
    models = {
        "Linear_Regression": LinearRegression(),
        "Random_Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    }
    
    best_r2 = -float("inf")
    best_model_name = None
    
    # 4. Bucle de entrenamiento y registro en MLflow
    for model_name, model in models.items():
        # Iniciamos un run individual para cada modelo dentro del experimento
        with mlflow.start_run(run_name=model_name):
            print(f"Entrenando {model_name}...")
            
            # Entrenar
            model.fit(X_train_scaled, y_train)
            
            # Predecir
            predictions = model.predict(X_test_scaled)
            
            # Evaluar
            mae, r2 = eval_metrics(y_test, predictions)
            print(f"{model_name} -> MAE: {mae:.4f}, R2: {r2:.4f}")
            
            # Registrar hiperparámetros específicos
            if hasattr(model, "n_estimators"):
                mlflow.log_param("n_estimators", model.get_params()["n_estimators"])
            if hasattr(model, "max_depth"):
                mlflow.log_param("max_depth", model.get_params()["max_depth"])
                
            # Registrar métricas en MLflow
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            
            # Registrar el artefacto del modelo (utilizando el sabor correcto)
            if model_name == "XGBoost":
                mlflow.xgboost.log_model(model, artifact_path="model")
            else:
                mlflow.sklearn.log_model(model, artifact_path="model")
                
            # Identificar el mejor modelo basado en R²
            if r2 > best_r2:
                best_r2 = r2
                best_model_name = model_name

    print(f"\nEl mejor modelo es: {best_model_name} con un R2 de {best_r2:.4f}")

if __name__ == "__main__":
    run_training()
