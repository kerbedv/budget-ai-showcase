# server/loader.py
from __future__ import annotations

import os, json, tempfile, shutil, logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import joblib

logger = logging.getLogger(__name__)

# --- ENV knobs (all optional) ---
# Private bucket that stores your artifacts
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "").strip()
# Optional prefix inside bucket, e.g. "budget-ai"
MODEL_PREFIX = os.getenv("MODEL_PREFIX", "").strip()
# Local cache root inside container
ARTIFACTS_DIR = Path(os.getenv("ARTIFACTS_DIR", "/app/artifacts"))
# Explicit GCP project (rarely needed on Cloud Run; ADC usually infers)
GCP_PROJECT = os.getenv("GCP_PROJECT", "").strip()


# ------------------------------
# GCS helpers (ADC-based)
# ------------------------------
def _gcs_download_dir(bucket_name: str, prefix: str, dest: Path, project: Optional[str] = None) -> int:
    """
    Download all objects under gs://bucket/prefix → dest/**.
    Returns number of files downloaded.
    Requires google-cloud-storage and ADC.
    """
    try:
        from google.cloud import storage
    except Exception as e:
        raise RuntimeError("google-cloud-storage is not installed. Add it to requirements.txt") from e

    client = storage.Client(project=project) if project else storage.Client()
    bucket = client.bucket(bucket_name)
    norm_prefix = prefix.lstrip("/")  # avoid leading slash issues

    n = 0
    for blob in client.list_blobs(bucket, prefix=norm_prefix):
        rel = blob.name[len(norm_prefix):].lstrip("/")
        # skip "directory markers"
        if not rel or rel.endswith("/"):
            continue
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(target))
        n += 1
    return n


def sync_all_from_gcs(local_root: Path = ARTIFACTS_DIR) -> int:
    """
    Optional: one-shot mirror of the configured {MODEL_BUCKET}/{MODEL_PREFIX}
    into local_root. Useful to warm cache at container start.
    """
    if not MODEL_BUCKET:
        logger.info("sync_all_from_gcs: MODEL_BUCKET not set; skipping.")
        return 0

    prefix = MODEL_PREFIX
    logger.info("Syncing GCS → local: bucket=%s prefix=%s -> %s", MODEL_BUCKET, prefix, local_root)
    tmp = Path(tempfile.mkdtemp())
    try:
        n = _gcs_download_dir(MODEL_BUCKET, prefix, tmp, project=GCP_PROJECT if GCP_PROJECT else None)
        if n > 0:
            local_root.mkdir(parents=True, exist_ok=True)
            for p in tmp.rglob("*"):
                if p.is_file():
                    rel = p.relative_to(tmp)
                    dst = local_root / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, dst)
            logger.info("GCS sync complete: %d files", n)
        else:
            logger.warning("GCS sync: no files found under %s/%s", MODEL_BUCKET, prefix)
        return n
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------
# Versioned bundle helpers
# Layout expected:
#   <local_root>/<model_name>/v1.2.3/model.joblib
#   <local_root>/<model_name>/v1.2.3/metadata.json
# ------------------------------
def _latest_version_dir(local_root: Path, model_name: str) -> Optional[Path]:
    root = local_root / model_name
    if not root.exists():
        return None
    cands = [d for d in root.iterdir() if d.is_dir() and d.name.startswith("v")]
    if not cands:
        return None

    def _parse_v(p: Path):
        s = p.name.lstrip("v")
        parts = (s.split(".") + ["0", "0", "0"])[:3]
        try:
            return tuple(int(x) for x in parts)
        except Exception:
            return (0, 0, 0)

    cands.sort(key=_parse_v)
    return cands[-1]


def _maybe_pull_bundle_from_gcs(local_dir: Path, model_name: str, version: str) -> bool:
    """
    Download one versioned bundle if missing locally.
    Bucket path: /{MODEL_PREFIX?}/{model_name}/v{version}/**
    """
    if not MODEL_BUCKET:
        return False

    prefix_parts = [MODEL_PREFIX, model_name, f"v{version}"]
    blob_prefix = "/".join(s for s in prefix_parts if s)

    tmp = Path(tempfile.mkdtemp())
    try:
        n = _gcs_download_dir(MODEL_BUCKET, blob_prefix, tmp, project=GCP_PROJECT if GCP_PROJECT else None)
        if n > 0:
            local_dir.mkdir(parents=True, exist_ok=True)
            for p in tmp.rglob("*"):
                if p.is_file():
                    rel = p.relative_to(tmp)
                    dst = local_dir / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, dst)
            logger.info("Pulled bundle from GCS: bucket=%s prefix=%s → %s (files=%d)",
                        MODEL_BUCKET, blob_prefix, local_dir, n)
            return True
        logger.warning("Bundle not found in GCS: bucket=%s prefix=%s", MODEL_BUCKET, blob_prefix)
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def load_bundle(name: str, version: Optional[str], local_root: Path = ARTIFACTS_DIR) -> Tuple[Any, Dict[str, Any], Path]:
    """
    Load a specific bundle (name + version). If local missing and MODEL_BUCKET is set,
    attempts to pull from GCS.

    Returns: (pipeline, metadata, bundle_dir)
    Raises: FileNotFoundError if not found locally nor in GCS
    """
    if version is None:
        vdir = _latest_version_dir(local_root, name)
        if not vdir:
            raise FileNotFoundError(f"No versions found for {name} under {local_root}")
        bundle_dir = vdir
    else:
        bundle_dir = local_root / name / f"v{version}"

    model_path = bundle_dir / "model.joblib"
    meta_path = bundle_dir / "metadata.json"

    # Try local first
    if not (model_path.exists() and meta_path.exists()):
        # Try GCS if configured
        v_clean = (version or "").lstrip("v")
        pulled = _maybe_pull_bundle_from_gcs(bundle_dir, name, v_clean)
        if not pulled or not (model_path.exists() and meta_path.exists()):
            raise FileNotFoundError(f"Missing bundle: {bundle_dir} (tried local and GCS)")

    pipe = joblib.load(model_path)
    meta = json.loads(meta_path.read_text())
    return pipe, meta, bundle_dir


def load_latest_or_env(name: str, local_root: Path = ARTIFACTS_DIR, envvar: Optional[str] = None) -> Tuple[Any, Dict[str, Any], Path]:
    """
    Convenience:
      - If ENV var is set (e.g., VER_XGB_SPEND=1.0.0), loads that version.
      - Else loads latest available local version (no GCS listing).
    Returns: (pipeline, metadata, bundle_dir)
    """
    ver = os.getenv(envvar, "").strip() if envvar else ""
    if ver:
        return load_bundle(name, ver, local_root)
    vdir = _latest_version_dir(local_root, name)
    if not vdir:
        raise FileNotFoundError(f"No local versions found for {name}; set {envvar} or warm local cache via GCS.")
    pipe = joblib.load(vdir / "model.joblib")
    meta = json.loads((vdir / "metadata.json").read_text())
    return pipe, meta, vdir


# ------------------------------
# Optional: simple filename-based loader (non-versioned)
# For quick POC where you dump single files like:
#   artifacts/xgb_spend.joblib, artifacts/kmeans.joblib, artifacts/prophet_v1.joblib
# ------------------------------
def load_simple_models(local_root: Path = ARTIFACTS_DIR, names: tuple[str, ...] = ("xgb_spend", "kmeans", "prophet_v1")) -> Dict[str, Any]:
    """
    Loads simple, non-versioned .joblib artifacts by conventional names.
    If MODEL_BUCKET is set and files missing, mirrors full prefix first.
    """
    local_root.mkdir(parents=True, exist_ok=True)
    loaded: Dict[str, Any] = {}

    # If any is missing and GCS is configured, try to warm the entire tree once
    missing = [n for n in names if not (local_root / f"{n}.joblib").exists()]
    if missing and MODEL_BUCKET:
        logger.info("Some simple artifacts missing locally (%s). Attempting GCS sync…", ", ".join(missing))
        sync_all_from_gcs(local_root=local_root)

    for n in names:
        p = local_root / f"{n}.joblib"
        if p.exists():
            loaded[n] = joblib.load(p)
        else:
            logger.warning("Simple artifact not found: %s", p)
    return loaded
