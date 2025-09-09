from pathlib import Path
from datetime import datetime
import hashlib, json, joblib, sys

ART_DIR = Path("artifacts")

def save_bundle(model_name: str, version: str, pipe, feature_names: list[str], extra_meta: dict | None = None):
    outdir = ART_DIR / model_name / f"v{version}"
    outdir.mkdir(parents=True, exist_ok=True)

    model_path = outdir / "model.joblib"
    joblib.dump(pipe, model_path)              # save
    _ = joblib.load(model_path)                # quick load to verify integrity

    meta = {
        "name": model_name,
        "version": version,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "feature_names": list(feature_names),  # ensure order is preserved
        "artifact_hash_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "sklearn_version": __import__("sklearn").__version__,
        "xgboost_version": __import__("xgboost").__version__,
        "python_version": sys.version.split()[0],
    }
    if extra_meta:
        meta.update(extra_meta)

    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"✅ Saved → {outdir}")
    return outdir, meta
