import json
import joblib
import pandas as pd
from ml import config, train

_cache = {"bundle": None}

def _bundle():
    if _cache["bundle"] is None and config.MODEL_PATH.exists():
        _cache["bundle"] = joblib.load(config.MODEL_PATH)
    return _cache["bundle"]

def get_data(limit: int = 2000) -> dict:
    df = pd.read_csv(config.PROCESSED_DATA_PATH) if config.PROCESSED_DATA_PATH.exists() else train.clean(train.load_raw())
    return {"total_rows": len(df), "rows": json.loads(df.head(limit).to_json(orient="records"))}

def get_metrics() -> dict:
    if not config.METRICS_PATH.exists(): return None
    return json.loads(config.METRICS_PATH.read_text())

def train_model() -> dict:
    train.run()
    _cache["bundle"] = None
    return get_metrics()

def make_prediction(features: dict) -> dict:
    bundle = _bundle()
    if bundle is None: raise RuntimeError("Model not trained.")
    
    cols = bundle["input_columns"]
    X = pd.DataFrame([{c: features[c] for c in cols}])
    pipe = bundle["pipeline"]
    
    p = float(pipe.predict_proba(X)[0, 1])
    return {
        "prediction": 1 if p >= 0.5 else 0,
        "probabilities": {"0": round(1 - p, 4), "1": round(p, 4)}
    }