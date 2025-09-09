from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment

from common import load_base, add_derived_features, version
from bundle_utils import save_bundle

ART = Path("artifacts")
MODEL_NAME = "kmeans_spend"
CSV_PATH = "synthetic_data_synth_real_combined.csv"

# Human-readable, stable labels
CLUSTER_LABELS: Dict[int, str] = {
    0: "Casual Spender",
    1: "Out and About Spender",
    2: "Fun-First Spender",
    3: "Early Riser Spender",
    4: "Financial Balancer",
    5: "Budget Guardian",
}

FEATURES = [
    "Savings_Rate",
    "Entertainment_Percentage",
    "Weekend_Percentage",
    "Average_Spending_Per_Day_Rate",
    "Food_Morning_Percentage",
    "Evening_Spending_Percentage",
]

def _parse_ver(v: str) -> Tuple[int, int, int]:
    v = v.lstrip("v")
    parts = (v.split(".") + ["0", "0", "0"])[:3]
    return tuple(int(x) for x in parts)

def _find_previous_dir(cur_version: str) -> Optional[Path]:
    root = ART / MODEL_NAME
    if not root.exists():
        return None
    cur_tuple = _parse_ver(cur_version)
    candidates = []
    for d in root.iterdir():
        if not d.is_dir() or not d.name.startswith("v"):
            continue
        t = _parse_ver(d.name)
        if t < cur_tuple:
            candidates.append((t, d))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]

# --------------------------------------------------------------------
# New: function entry so the orchestrator can call this with a prepared DF
# --------------------------------------------------------------------
def train_kmeans_from_df(df_cluster: pd.DataFrame, ver: str) -> Path:
    """
    Train and save kmeans_spend from a prebuilt DataFrame (clustering variant).

    Assumes:
      - df_cluster already has the 'clustering' derived features.
    Returns:
      outdir (Path) to artifacts/kmeans_spend/v<ver>
    """
    # Schema check + sanitize
    missing = [c for c in FEATURES if c not in df_cluster.columns]
    if missing:
        raise ValueError(f"KMeans expects {FEATURES}, missing {missing}")

    X = (
        df_cluster.loc[:, FEATURES]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    # Fit pipeline deterministically
    pipe = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        KMeans(n_clusters=6, n_init=10, random_state=42),  # pinned for reproducibility
    )
    pipe.fit(X)

    # Align cluster IDs to previous version (stable semantics)
    k_new = pipe.named_steps["kmeans"].cluster_centers_
    id_map = {i: i for i in range(len(k_new))}
    prev_dir = _find_previous_dir(f"v{ver}")
    if prev_dir is not None:
        pipe_prev = joblib.load(prev_dir / "model.joblib")
        k_prev = pipe_prev.named_steps["kmeans"].cluster_centers_
        D = pairwise_distances(k_prev, k_new)
        row_ind, col_ind = linear_sum_assignment(D)
        id_map = {int(new): int(prev) for prev, new in zip(row_ind, col_ind)}  # NEW -> STABLE

    # Save bundle + rich metadata
    extra_meta = {
        "model_type": "kmeans",
        "cluster_id_map": {str(k): int(v) for k, v in id_map.items()},
        "cluster_labels": {str(k): v for k, v in CLUSTER_LABELS.items()},
        "n_clusters": int(pipe.named_steps["kmeans"].n_clusters),
    }
    outdir, _meta = save_bundle(MODEL_NAME, ver, pipe, FEATURES, extra_meta=extra_meta)
    print(f"✅ Saved → {outdir}")
    return outdir


def main():
    ver = version()
    outdir = ART / MODEL_NAME / f"v{ver}"
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) Load & features (match notebook)
    df = load_base(CSV_PATH)
    df = add_derived_features(df, variant="clustering")

    # 2) Call the new function entry
    out = train_kmeans_from_df(df, ver)
    print(f"✅ Saved → {out}")

if __name__ == "__main__":
    main()
