import pandas as pd
import joblib
import sklearn
import os
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT_DIR, "data", "Loan_default.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "final_rf_pipeline_with_metadata.joblib")

def run():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=['Default', 'LoanID'])
    y = df['Default']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    numerical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

    rf_params = {'n_estimators': 200, 'max_depth': 10, 'class_weight': 'balanced', 'random_state': 42}

    rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(**rf_params))])

    print("Training Random Forest...")
    rf_pipeline.fit(X_train, y_train)

    joblib.dump((rf_pipeline, {"scikit_learn_version": sklearn.__version__}), MODEL_PATH)
    print(f"Model saved successfully to {MODEL_PATH}")

if __name__ == "__main__":
    run()