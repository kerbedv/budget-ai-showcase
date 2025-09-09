from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Union

import joblib
import numpy as np
import pandas as pd
import os
import json

# =========================
# Paths & Data Root
# =========================
# Repo root = parent of /training
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Default training data folder = <repo>/training/data; can override with DATA_DIR
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "training" / "data")).resolve()

def data_path(filename: Union[str, Path]) -> Path:
    """
    Resolve a CSV filename to an absolute path.
    Priority:
      1) Absolute paths are returned as-is.
      2) If a relative path exists in CWD, use it.
      3) Otherwise, look under DATA_DIR/<filename>.
    """
    p = Path(filename)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return (DATA_DIR / p.name).resolve()

# =========================
# Core helpers
# =========================
def _safe_div(numer, denom) -> pd.Series:
    # Always return a Series; works for scalars & arrays
    numer_s = pd.Series(numer)
    denom_s = pd.Series(denom)

    # Treat 0 and NaN/Inf as missing
    denom_s = denom_s.replace([0, np.inf, -np.inf], np.nan)
    out = numer_s / denom_s
    return out.fillna(0.0)


def _ensure_cols(df: pd.DataFrame, cols: Iterable[str], fill: float = 0.0) -> pd.DataFrame:
    """
    Make sure required columns exist; if missing, create and fill with `fill`.
    """
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = fill
    return df

def load_base(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load CSV from training/data (or DATA_DIR override) and normalize column names.
    """
    csv_path = data_path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found at {csv_path}. "
            f"Looked under DATA_DIR={DATA_DIR}. "
            "Set DATA_DIR env var to override."
        )
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    return df

# =========================
# Feature Engineering
# =========================
def add_derived_features(
    df: pd.DataFrame,
    variant: str = "xgb",
    days_in_month: Optional[int] = 30,
) -> pd.DataFrame:
    """
    Compute the derived features used in your notebooks.

    Parameters
    ----------
    variant : 'xgb' | 'clustering'
        - 'clustering' uses Entertainment_Percentage = (Entertainment + Food + Gifts) / Income
        - 'xgb'       uses Entertainment_Percentage = (Entertainment + Gifts) / Income,
                      and adds separate Food_Percentage
    days_in_month : int or None
        If provided, uses a fixed constant (your notebooks commonly used 30).
        If None and df has a 'ds' datetime column, will infer month length per row.

    Notes
    -----
    Creates the following columns (superset):
      - Savings_Rate
      - Entertainment_Percentage
      - Weekend_Percentage
      - Days_in_Month
      - Average_Spending_Per_Day
      - Average_Spending_Per_Day_Rate
      - Food_Morning_Percentage
      - Evening_Spending_Percentage
      - Spending_Rate
      - Spending_Intensity
      - Food_Percentage
      - Credit_Rate
      - Housing_Rate
      - Health_Rate
    """
    df = df.copy()

    # Ensure raw columns exist (fill zeros if missing)
    base_cols = [
        "Income", "Savings", "Investments", "Entertainment", "Food", "Gifts",
        "Weekend_Spend", "Total_Spending", "Food_Morning_Spend",
        "Entertainment_Evening_Spend", "Credit_Payment", "Housing", "Health",
    ]
    df = _ensure_cols(df, base_cols, fill=0.0)

    # Savings_Rate
    df["Savings_Rate"] = _safe_div(df["Savings"] + df["Investments"], df["Income"])

    # Entertainment_Percentage (variant-dependent)
    if variant == "clustering":
        df["Entertainment_Percentage"] = _safe_div(
            df["Entertainment"] + df["Food"] + df["Gifts"], df["Income"]
        )
    else:  # 'xgb' default
        df["Entertainment_Percentage"] = _safe_div(
            df["Entertainment"] + df["Gifts"], df["Income"]
        )
        # Separate Food_Percentage for xgb
        df["Food_Percentage"] = _safe_div(df["Food"], df["Income"])

    # Weekend_Percentage
    df["Weekend_Percentage"] = _safe_div(df["Weekend_Spend"], df["Income"])

    # Days_in_Month
    if days_in_month is not None:
        df["Days_in_Month"] = int(days_in_month)
    else:
        # Infer per-row from 'ds' if present; fallback to 30
        if "ds" in df.columns:
            ds = pd.to_datetime(df["ds"], errors="coerce")
            inferred = ds.dt.days_in_month.fillna(30).astype(int)
            df["Days_in_Month"] = inferred
        else:
            df["Days_in_Month"] = 30

    # Average daily spending + rate
    df["Average_Spending_Per_Day"] = _safe_div(df["Total_Spending"], df["Days_in_Month"])
    df["Average_Spending_Per_Day_Rate"] = _safe_div(
        df["Average_Spending_Per_Day"], df["Income"]
    )

    # Morning / Evening components
    df["Food_Morning_Percentage"] = _safe_div(df["Food_Morning_Spend"], df["Income"])
    df["Evening_Spending_Percentage"] = _safe_div(
        df["Entertainment_Evening_Spend"], df["Income"]
    )

    # Spending_Rate + Intensity
    df["Spending_Rate"] = _safe_div(df["Total_Spending"], df["Income"])
    df["Spending_Intensity"] = (
        df["Average_Spending_Per_Day_Rate"]
        + df["Entertainment_Percentage"]
        + df["Weekend_Percentage"]
    )

    # Other ratios used by XGBs
    df["Credit_Rate"] = _safe_div(df["Credit_Payment"], df["Income"])
    df["Housing_Rate"] = _safe_div(df["Housing"], df["Income"])
    df["Health_Rate"] = _safe_div(df["Health"], df["Income"])

    # Guarantee all expected columns exist (even if variant produced some conditionally)
    expected = [
        "Savings_Rate", "Entertainment_Percentage", "Weekend_Percentage",
        "Days_in_Month", "Average_Spending_Per_Day", "Average_Spending_Per_Day_Rate",
        "Food_Morning_Percentage", "Evening_Spending_Percentage", "Spending_Rate",
        "Spending_Intensity", "Food_Percentage", "Credit_Rate", "Housing_Rate",
        "Health_Rate",
    ]
    df = _ensure_cols(df, expected, fill=0.0)

    # Type normalization (floats)
    for c in expected:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    return df

# =========================
# Behavior_Cluster (via KMeans bundle)
# =========================
KMEANS_FEATURES = [
    "Savings_Rate",
    "Entertainment_Percentage",
    "Weekend_Percentage",
    "Average_Spending_Per_Day_Rate",
    "Food_Morning_Percentage",
    "Evening_Spending_Percentage",
]

def load_kmeans_bundle(bundle_dir: Union[str, Path]):
    """
    Load a saved KMeans pipeline bundle from:
      artifacts/kmeans_spend/vX.Y.Z/{model.joblib, metadata.json}
    Returns (pipe, meta_dict).
    """
    bundle_dir = Path(bundle_dir)
    pipe = joblib.load(bundle_dir / "model.joblib")
    meta = json.loads((bundle_dir / "metadata.json").read_text())
    return pipe, meta

def add_behavior_cluster(
    df: pd.DataFrame,
    kmeans_pipe_or_path: Union[str, Path, object],
    ensure_variant: str = "clustering",
) -> pd.DataFrame:
    df = add_derived_features(df, variant=ensure_variant).copy()

    if isinstance(kmeans_pipe_or_path, (str, Path)):
        pipe, meta = load_kmeans_bundle(kmeans_pipe_or_path)
        feats = meta.get("feature_names", KMEANS_FEATURES)
    else:
        pipe = kmeans_pipe_or_path
        feats = KMEANS_FEATURES  # best we can do without metadata

    # Ensure features exist & ordered to the bundle schema
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"KMeans bundle expects features {feats}, missing: {missing}")

    # Build X as a DataFrame (names preserved), coerce to numeric, sanitize inf/NaN
    X = (
        df.loc[:, feats]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    df["Behavior_Cluster"] = pipe.predict(X).astype(int)
    return df


# =========================
# Versioning
# =========================
def version() -> str:
    """
    Returns the model bundle version for this training run.
    Prefer setting the MODEL_VERSION env var when invoking trainers.
    Falls back to 1.0.0 if unset.
    """
    return os.getenv("MODEL_VERSION", "1.0.0")
