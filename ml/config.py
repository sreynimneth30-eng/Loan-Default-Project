from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "loan_default.csv"
PROCESSED_DATA_PATH = ROOT_DIR / "data" / "processed" / "clean_loan.csv"
MODEL_PATH = ROOT_DIR / "models" / "model.joblib"
METRICS_PATH = ROOT_DIR / "models" / "metrics.json"
FIGURES_DIR = ROOT_DIR / "reports" / "figures"

# Problem definition
TASK_TYPE = "classification"
TARGET_COLUMN = "Default"
POSITIVE_CLASS = 1
NEGATIVE_CLASS = 0
DROP_COLUMNS = ["LoanID"]

# Training
TEST_SIZE = 0.2
CV_FOLDS = 5
RANDOM_STATE = 42
PRIMARY_METRIC = "f1"