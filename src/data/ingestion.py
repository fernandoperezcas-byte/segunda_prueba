import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def load_raw_data() -> pd.DataFrame:
    df = fetch_california_housing(as_frame=True).frame
    df = df.rename(columns={'MedHouseVal': 'target'}) # Renombrar la variable objetivo para mayor claridad
    df = df[df['AveRooms'] <= 10]
    df = df[df['AveOccup'] <= 10]
    df = df[df['Population'] <= 6000]
    df = df[df['MedInc'] <= 11]
    return df

def split_data(df: pd.DataFrame, target_column: str = 'target', test_size: float = 0.25, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Prueba rápida del módulo
    print("Iniciando ingesta de datos...")
    data = load_raw_data()
    X_tr, X_te, y_tr, y_te = split_data(data)
    print(f"Datos cargados. Dimensiones de entrenamiento: {X_tr.shape}, Prueba: {X_te.shape}")
