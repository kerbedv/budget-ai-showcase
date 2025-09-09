from typing import Dict, Any
import numpy as np

from mapping import to_ordered_array


def _predict_one(name: str, models: dict, metas: dict, data: Dict[str, float]) -> float:
    """
    Predict with a single submodel using its schema from metadata.
    Applies optional value_range clipping if present in metadata.
    """
    if name not in models or name not in metas:
        raise KeyError(f"missing_model_or_meta::{name}")

    feats = metas[name]["feature_names"]
    x = np.array([to_ordered_array(data, feats)], dtype=float)
    y = float(models[name].predict(x)[0])

    # Optional clip to business-valid range if provided
    vr = metas[name].get("value_range")
    if isinstance(vr, (list, tuple)) and len(vr) == 2:
        y = float(np.clip(y, float(vr[0]), float(vr[1])))

    return y


def ensure_cluster(kmeans_meta: Dict[str, Any], kmeans_pipe, data: Dict[str, float]) -> int:
    """
    Compute Behavior_Cluster using the trained KMeans bundle.
    """
    feats = kmeans_meta["feature_names"]
    x = np.array([to_ordered_array(data, feats)], dtype=float)
    return int(kmeans_pipe.predict(x)[0])


def predict_meta_features(models: dict, metas: dict, data: Dict[str, float]) -> Dict[str, float]:
    """
    Compute meta-features needed by xgb_savings_goal at serving time.
    Emits keys named OOF_* to match the classifier training schema.

    Returns:
        dict with:
          - OOF_Spending_Rate
          - OOF_Entertainment_Percentage
          - OOF_Housing_Rate
          - OOF_Credit_Rate
          - OOF_Health_Rate
          - OOF_Food_Percentage
    """
    out: Dict[str, float] = {}

    # Map submodel bundle name -> target label used in training
    pairs = [
        ("xgb_spending_rate",            "Spending_Rate"),
        ("xgb_entertainment_percentage", "Entertainment_Percentage"),
        ("xgb_housing_rate",             "Housing_Rate"),
        ("xgb_credit_rate",              "Credit_Rate"),
        ("xgb_health_rate",              "Health_Rate"),
        ("xgb_food_percentage",          "Food_Percentage"),
    ]

    for bundle_name, target_label in pairs:
        out[f"OOF_{target_label}"] = _predict_one(bundle_name, models, metas, data)

    return out
