import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        X_train_scaled = self.scaler.fit_transform(X_train)
        # Mantenemos los nombres de las columnas para conservar la estructura
        return pd.DataFrame(X_train_scaled, columns=X_train.columns)

    def transform(self, X_test: pd.DataFrame) -> pd.DataFrame:
        X_test_scaled = self.scaler.transform(X_test)
        return pd.DataFrame(X_test_scaled, columns=X_test.columns)

if __name__ == "__main__":
    # Prueba rápida del pipeline de procesamiento
    from data_ingestion import load_raw_data, split_data
    
    print("Iniciando pipeline de procesamiento...")
    data = load_raw_data()
    X_train, X_test, y_train, y_test = split_data(data)
    
    processor = DataProcessor()
    X_train_scaled = processor.fit_transform(X_train)
    X_test_scaled = processor.transform(X_test)
    
    print("¡Procesamiento exitoso!")
    print(f"Media de la primera columna escalada (debería ser ~0): {X_train_scaled.iloc[:, 0].mean():.4f}")
