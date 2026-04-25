import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

DATA_DIR = "../data"

def load_data(train_path: str = f"{DATA_DIR}/fraudTrain.csv", test_path: str = f"{DATA_DIR}/fraudTest.csv"):
    """Load train and test CSV files into DataFrames."""
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Basic preprocessing:
    - Parse transaction datetime
    - Extract hour, dayofweek, month
    - Encode categorical columns with OrdinalEncoder
    - Drop columns that are identifiers or highly sparse
    """
    df = df.copy()
    # Drop unnamed index column if present
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
    # Parse datetime
    if "trans_date_trans_time" in df.columns:
        df["trans_dt"] = pd.to_datetime(df["trans_date_trans_time"], errors="coerce")
        df["hour"] = df["trans_dt"].dt.hour
        df["dayofweek"] = df["trans_dt"].dt.dayofweek
        df["month"] = df["trans_dt"].dt.month
        df.drop(columns=["trans_date_trans_time", "trans_dt"], inplace=True)
    # Fill missing numeric with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    # Encode categoricals
    cat_cols = df.select_dtypes(include=[object]).columns.tolist()
    if cat_cols:
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df[cat_cols] = encoder.fit_transform(df[cat_cols])
    return df

def get_train_test_split(test_size: float = 0.2, random_state: int = 42):
    train_df, _ = load_data()
    train_df = preprocess(train_df)
    X = train_df.drop(columns=["is_fraud"])
    y = train_df["is_fraud"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
