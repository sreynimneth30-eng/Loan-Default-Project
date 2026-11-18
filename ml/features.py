import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from ml import config

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"LoanAmount", "Income"} <= set(df.columns):
        df["loan_to_income"] = df["LoanAmount"] / (df["Income"] + 1)
    return df

def get_input_columns(df: pd.DataFrame) -> list[str]:
    skip = {config.TARGET_COLUMN, *config.DROP_COLUMNS}
    return [c for c in df.columns if c not in skip]

def build_preprocessor(X_sample: pd.DataFrame) -> Pipeline:
    numeric = X_sample.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X_sample.columns if c not in numeric]
    
    columns = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    return Pipeline([("add_features", FunctionTransformer(add_features)), ("columns", columns)])