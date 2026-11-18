import pandas as pd
from ml import config

def load_raw() -> pd.DataFrame:
    return pd.read_csv(config.RAW_DATA_PATH)

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().drop_duplicates()
    df = df.drop(columns=config.DROP_COLUMNS, errors="ignore")
    return df

if __name__ == "__main__":
    raw = load_raw()
    clean_df = clean(raw)
    config.PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(config.PROCESSED_DATA_PATH, index=False)
    print("Saved cleaned data!")