from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

from common import load_base, add_derived_features, add_behavior_cluster, version
from bundle_utils import save_bundle

ART = Path("artifacts")
MODEL_NAME = "xgb_spending_rate"
TARGET = "Spending_Rate"

# Final training features (requires Behavior_Cluster)
FEATURES = [
    "Income","Savings_Rate","Entertainment_Percentage","Weekend_Percentage",
    "Food_Morning_Percentage","Evening_Spending_Percentage","Spending_Intensity",
    "Food_Percentage","Credit_Rate","Behavior_Cluster"
]

# Business-valid range for serving (clip prediction to this)
VALUE_RANGE = [0.0, 1.5]  # use [0.0, 1.0] if your real data can't exceed income

CSV_PATH = "synthetic_data_synth_real_combined.csv"


def _xgb():
    # Deterministic & CPU-friendly
    return XGBRegressor(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.9,
        random_state=42,
        tree_method="hist",
        n_jobs=1,
    )

def make_preprocessor(feature_list):
    return ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), feature_list)],
        remainder="drop"
    )

def make_model(feature_list):
    return make_pipeline(make_preprocessor(feature_list), _xgb())

def compute_cv_rmse(X, y, features, k=5, seed=42):
    rmses = []
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    for tr, va in kf.split(X):
        pipe_fold = make_model(features)  # fresh preprocessor + estimator
        pipe_fold.fit(X.iloc[tr], y[tr])
        p = pipe_fold.predict(X.iloc[va])
        rmses.append(float(np.sqrt(mean_squared_error(y[va], p))))
    return float(np.mean(rmses)), float(np.std(rmses))


# --------------------------------------------------------------------
# New: function entry so an orchestrator can call this with a prepared DF
# --------------------------------------------------------------------
def train_xgb_spending_rate_from_df(df_xgb, ver: str):
    """
    Train and save xgb_spending_rate from a prebuilt DataFrame.

    Assumes:
      - df_xgb already has all derived features for the 'xgb' variant
      - df_xgb includes a valid 'Behavior_Cluster' column (int)
    """
    # 1) Build training matrix
    missing = [c for c in FEATURES if c not in df_xgb.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X = df_xgb[FEATURES]
    y = df_xgb[TARGET].astype(float).values

    # 2) Fit full pipeline
    pipe = make_model(FEATURES)
    pipe.fit(X, y)

    # 3) 5-fold CV RMSE (metadata only)
    cv_mean, cv_std = compute_cv_rmse(X, y, FEATURES, k=5, seed=42)

    # 4) Save bundle + rich metadata
    extra_meta = {
        "model_type": "regressor",
        "target": TARGET,
        "cv_rmse_mean": cv_mean,
        "cv_rmse_std":  cv_std,
        "value_range": VALUE_RANGE,                         # serve-time clipping
        "kmeans_source": f"artifacts/kmeans_spend/v{ver}",  # provenance
    }
    save_bundle(MODEL_NAME, ver, pipe, FEATURES, extra_meta=extra_meta)

    outdir = ART / MODEL_NAME / f"v{ver}"
    return cv_mean, cv_std, outdir


def main():
    ver = version()
    outdir = ART / MODEL_NAME / f"v{ver}"
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load & derive features (XGB variant)
    base = load_base(CSV_PATH)
    df_xgb = add_derived_features(base, variant="xgb")

    # 2) Ensure Behavior_Cluster from the trained KMeans bundle (same version)
    if "Behavior_Cluster" not in df_xgb.columns or df_xgb["Behavior_Cluster"].isna().any():
        kmeans_dir = ART / "kmeans_spend" / f"v{ver}"
        if not kmeans_dir.exists():
            print(
                f"[ERROR] Missing KMeans bundle at: {kmeans_dir}\n"
                f"Train it first: python training/train_kmeans.py",
                file=sys.stderr
            )
            sys.exit(2)
        df_cluster = add_behavior_cluster(base.copy(), kmeans_dir, ensure_variant="clustering")
        df_xgb["Behavior_Cluster"] = (
            df_cluster["Behavior_Cluster"].astype(int).reindex(df_xgb.index)
        )

    # 3) Call the new function entry
    cv_mean, cv_std, out = train_xgb_spending_rate_from_df(df_xgb, ver)
    print(f"✅ Saved → {out} (cv_rmse_mean={cv_mean:.4f}, cv_rmse_std={cv_std:.4f})")


if __name__ == "__main__":
    main()
