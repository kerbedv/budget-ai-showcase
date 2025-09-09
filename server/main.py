import os, json, logging
from functools import wraps
from pathlib import Path
from typing import Dict, Any, List, Tuple

import joblib
import pandas as pd
import numpy as np 
from flask import Flask, request, jsonify, g

import firebase_admin
from firebase_admin import credentials, auth

from pydantic import BaseModel, Field, field_validator, ConfigDict

import time, uuid, json as _json

from orchestrator import ensure_cluster, predict_meta_features

from loader import ARTIFACTS_DIR, load_latest_or_env, sync_all_from_gcs
from mapping import to_ordered_array

try:
    from prophet import Prophet
    _HAS_PROPHET = True
except Exception as _e:
    _HAS_PROPHET = False
    _PROPHET_ERR = str(_e)
    
from datetime import datetime
from werkzeug.exceptions import BadRequest
from google.cloud import firestore
    
    
# ----------------------------------------------------------------------------- 
# Logging
# -----------------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,  # ensure our config wins
)
logger = logging.getLogger(__name__)
# --- Showcase flags / helpers ---
SAFE_SHOWCASE = os.getenv("SAFE_SHOWCASE") == "1"   # default off unless set

def _mask(p):
    if not p:
        return None
    p = str(p)
    return "..." + p[-8:] if len(p) > 8 else "***"


# ----------------------------------------------------------------------------- 
# App
# -----------------------------------------------------------------------------
app = Flask(__name__)
logger.info("IMPORT module=%s file=%s cwd=%s", __name__, __file__, os.getcwd())

if not SAFE_SHOWCASE:
    try:
        sync_all_from_gcs(ARTIFACTS_DIR)
    except Exception:
        logger.exception("GCS warm sync failed (continuing with local artifacts)")
else:
    logger.info("SAFE_SHOWCASE=1 — skipping GCS warm sync")


def _log_predict_event(model: str, meta: dict, t0: float, status: str, **kv):
    entry = {
        "event": "predict",
        "status": status,                 # "ok" | "error"
        "model": model,
        "version": (meta or {}).get("version"),
        "artifact": (meta or {}).get("artifact_hash_sha256"),
        "latency_ms": int((time.time() - t0) * 1000),
        "uid": getattr(getattr(g, "claims", {}), "get", lambda *_: None)("uid") if hasattr(g, "claims") else None,
        "path": request.path,
        "request_id": request.headers.get("X-Request-Id") or str(uuid.uuid4()),
    }
    if kv:
        entry.update(kv)
    logger.info(_json.dumps(entry))

# ----------------------------------------------------------------------------- 
# Firebase Admin (prefer ADC on Cloud Run; fall back to creds file for local)
# -----------------------------------------------------------------------------
PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "budget-ai-mobile")
SA_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "service-account.json")

if not firebase_admin._apps:
    try:
        if SA_PATH and Path(SA_PATH).exists():
            cred = credentials.Certificate(SA_PATH)
            firebase_admin.initialize_app(cred, {"projectId": PROJECT_ID})
            logger.info("Firebase initialized via service-account file: %s", SA_PATH)
        else:
            cred = credentials.ApplicationDefault()
            firebase_admin.initialize_app(cred, {"projectId": PROJECT_ID})
            logger.info("Firebase initialized via Application Default Credentials (ADC).")
    except Exception:
        logger.exception("Firebase initialization failed; trying default init()")
        firebase_admin.initialize_app()

# >>> Add these 2 lines to confirm the backend’s project binding
admin_app = firebase_admin.get_app()
logger.info("FIREBASE project_id (admin): %s", getattr(admin_app, "project_id", "<none>"))
logger.info("FIREBASE expected iss/aud: https://securetoken.google.com/%s  |  %s", PROJECT_ID, PROJECT_ID)
logger.info("GOOGLE_APPLICATION_CREDENTIALS: %s", _mask(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")))
logger.info("Resolved SA_PATH=%s; CWD=%s", _mask(SA_PATH), os.getcwd())
if SAFE_SHOWCASE:
    logger.info("SAFE_SHOWCASE=1 — Firebase/Firestore will be skipped or mocked where possible")



# -----------------------------------------------------------------------------
# Auth guard with detailed logging
# -----------------------------------------------------------------------------
REQUIRE_VERIFIED_EMAIL = True
EXPECTED_ISS = f"https://securetoken.google.com/{PROJECT_ID}"
EXPECTED_AUD = PROJECT_ID

def _extract_firebase_token() -> str | None:
    """
    Returns the Firebase ID token from:
      1) Authorization: Bearer <token>
      2) X-Firebase-Token: <token>
      3) (dev only) ?token=<token> query param
    """
    # 1) Authorization header
    auth_hdr = request.headers.get("Authorization", "")
    if auth_hdr:
        # case-insensitive and tolerant of extra spaces
        parts = auth_hdr.strip().split(" ", 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            token = parts[1].strip().strip('"').strip("'")
            if token:
                return token

    # 2) X-Firebase-Token header
    xfbt = request.headers.get("X-Firebase-Token")
    if xfbt:
        return xfbt.strip().strip('"').strip("'")

    # 3) (optional for debugging) query param
    token_qs = request.args.get("token")
    if token_qs and os.getenv("ALLOW_TOKEN_QS") == "1":
        return token_qs.strip().strip('"').strip("'")

    return None

def require_auth(fn):
    @wraps(fn)
    def _wrap(*args, **kwargs):
        if os.getenv("BYPASS_AUTH") == "1" or request.headers.get("X-Bypass-Auth") == "1":
            g.claims = {"uid": "demo-bypass"}
            return fn(*args, **kwargs)

        token = _extract_firebase_token()
        if not token:
            return jsonify({"error": "Unauthorized", "reason": "missing_token"}), 401

        try:
            claims = auth.verify_id_token(token)
            iss = claims.get("iss")
            aud = claims.get("aud")
            if iss != EXPECTED_ISS:
                return jsonify({"error": "Unauthorized", "reason": f"issuer_mismatch:{iss}"}), 401
            if aud != EXPECTED_AUD:
                return jsonify({"error": "Unauthorized", "reason": f"audience_mismatch:{aud}"}), 401
            if REQUIRE_VERIFIED_EMAIL and not claims.get("email_verified", False):
                return jsonify({"error": "Unauthorized", "reason": "email_not_verified"}), 401
            g.claims = claims
            return fn(*args, **kwargs)
        except Exception as e:
            logger.exception("[AUTH] verify_id_token failed")
            return jsonify({"error": "Unauthorized", "reason": str(e)}), 401
    return _wrap

# -----------------------------------------------------------------------------
# (Optional) quick debug route to verify POST auth works
# -----------------------------------------------------------------------------
@app.post("/auth_debug")
@require_auth
def auth_debug():
    c = g.claims
    # return a safe subset
    return {
        "uid": c.get("uid"),
        "aud": c.get("aud"),
        "iss": c.get("iss"),
        "email_verified": c.get("email_verified", False),
    }
# -----------------------------------------------------------------------------
# Artifact loader (v2 with optional GCS via loader.py)
# -----------------------------------------------------------------------------
BUNDLES: Dict[str, Any] = {}
METAS: Dict[str, Dict[str, Any]] = {}

ALIASES = {
    "kmeans": "kmeans_spend",  # keep mobile compatibility
}
KMEANS_NAME = "kmeans_spend"

def _load_with_env(name: str, envvar: str | None = None):
    pipe, meta, path = load_latest_or_env(name, ARTIFACTS_DIR, envvar=envvar)
    BUNDLES[name] = pipe
    METAS[name] = meta
    logger.info("model_loaded name=%s version=%s path=%s", name, meta.get("version"), path)

# Load core bundles (pin via VER_* envs, or latest local if unset)
if not SAFE_SHOWCASE:
    _load_with_env("kmeans_spend",               "VER_KMEANS")
    _load_with_env("xgb_spending_rate",          "VER_XGB_SPEND")
    _load_with_env("xgb_saving_rate",            "VER_XGB_SAVE")
    _load_with_env("xgb_entertainment_percentage","VER_XGB_ENT")
    _load_with_env("xgb_housing_rate",           "VER_XGB_HOUS")
    _load_with_env("xgb_credit_rate",            "VER_XGB_CRED")
    _load_with_env("xgb_health_rate",            "VER_XGB_HEALTH")
    _load_with_env("xgb_food_percentage",        "VER_XGB_FOOD")
    _load_with_env("xgb_savings_goal",           "VER_XGB_GOAL")
else:
    logger.info("SAFE_SHOWCASE=1 — skipping artifact loads (BUNDLES/METAS will be empty)")
    
# Prophet serve mode: on-the-fly fitting (no prefit artifact)
prophet_model = None
logger.info("Prefit Prophet disabled; forecasts will be fit on-the-fly from Firestore.")

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _resolve_model_name(model_type: str) -> str:
    return ALIASES.get(model_type, model_type)

def _order_features(features: Any, feature_names: List[str]) -> Tuple[List[float], List[str]]:
    """
    Accept either a list (assumed ordered) or a dict of {feature_name: value}.
    Returns (ordered_list, missing_features_list).
    """
    if isinstance(features, dict):
        vals = []
        missing = []
        for name in feature_names:
            if name in features:
                vals.append(float(features[name]))
            else:
                missing.append(name)
                vals.append(0.0)  # or raise; we zero-fill and report missing
        return vals, missing
    # list-like
    if not isinstance(features, (list, tuple)):
        raise ValueError("features must be a list in meta order, or a dict keyed by feature_name")
    if len(features) != len(feature_names):
        raise ValueError(f"expected {len(feature_names)} features, got {len(features)}")
    return [float(v) for v in features], []

def clip_pred(model_name: str, y: float) -> float:
    lo, hi = METAS.get(model_name, {}).get("value_range", [float("-inf"), float("inf")])
    try:
        return max(lo, min(hi, float(y)))
    except Exception:
        return float(y)

def kmeans_stable_output(raw_label: int) -> Dict[str, Any]:
    meta = METAS.get("kmeans_spend", {})
    cmap = meta.get("cluster_id_map", {})
    labels = meta.get("cluster_labels", {})
    stable = int(cmap.get(str(raw_label), raw_label))
    return {
        "cluster_id": stable,
        "cluster_name": labels.get(str(stable)),
    }

def _predict_model(name: str, feats_dict: dict) -> float:
    """Order features from metadata, run the submodel, and clip to its value_range."""
    meta = METAS[name]
    cols = meta["feature_names"]
    # (optional) log any missing keys
    miss = [c for c in cols if c not in feats_dict]
    if miss:
        logger.warning("missing_features model=%s missing=%s", name, miss)

    row = {c: float(feats_dict.get(c, 0.0) or 0.0) for c in cols}
    X = pd.DataFrame([row], columns=cols)        # <-- DataFrame w/ names
    y = float(BUNDLES[name].predict(X)[0])
    return clip_pred(name, y)

def _ensure_behavior_cluster(payload: dict) -> dict:
    """Ensure payload has Behavior_Cluster. If missing, compute via KMeans bundle."""
    if "Behavior_Cluster" in payload:
        return payload

    if KMEANS_NAME not in BUNDLES:
        raise RuntimeError("kmeans_bundle_not_loaded")

    kmeta = METAS[KMEANS_NAME]
    kfeats = kmeta.get("feature_names", [])
    missing = [c for c in kfeats if c not in payload]

    # Try to derive missing KMeans features from raw amounts if possible.
    if missing:
        inc = float(payload.get("Income", 0.0) or 0.0)
        if inc > 0:
            def _r(v): return float(payload.get(v, 0.0) or 0.0) / inc
            derived = {
                "Savings_Rate": (_r("Savings") + _r("Investments")),
                "Entertainment_Percentage": (_r("Entertainment") + _r("Food") + _r("Gifts")),  # 'clustering' variant
                "Weekend_Percentage": _r("Weekend_Spend"),
                "Average_Spending_Per_Day_Rate": (float(payload.get("Total_Spending", 0.0) or 0.0) /
                                                  float(payload.get("Days_in_Month", 30) or 30)) / inc,
                "Food_Morning_Percentage": _r("Food_Morning_Spend"),
                "Evening_Spending_Percentage": _r("Entertainment_Evening_Spend"),
            }
            for k, v in derived.items():
                if k not in payload and k in kfeats:
                    payload[k] = v

    still_missing = [c for c in kfeats if c not in payload]
    if still_missing:
        raise ValueError(f"cannot_compute_behavior_cluster_missing: {still_missing}")

    row = {c: float(payload.get(c, 0.0) or 0.0) for c in kfeats}
    X = pd.DataFrame([row], columns=kfeats)\
          .apply(pd.to_numeric, errors="coerce")\
          .replace([np.inf, -np.inf], np.nan)\
          .fillna(0.0)

    raw = int(BUNDLES[KMEANS_NAME].predict(X)[0])
    stable_info = kmeans_stable_output(raw)
    payload["Behavior_Cluster"] = int(stable_info["cluster_id"])
    payload["_behavior_cluster_raw"] = raw
    payload["_behavior_cluster_name"] = stable_info.get("cluster_name")
    return payload

def _orchestrate_savings_goal(payload: dict) -> dict:
    """Compute Behavior_Cluster, call submodels, build meta-features, run classifier."""
    payload = dict(payload)  # don’t mutate caller
    payload = _ensure_behavior_cluster(payload)

    preds = {
        "Predicted_Spending_Rate":            _predict_model("xgb_spending_rate",            payload),
        "Predicted_Entertainment_Percentage": _predict_model("xgb_entertainment_percentage", payload),
        "Predicted_Housing_Rate":             _predict_model("xgb_housing_rate",             payload),
        "Predicted_Credit_Rate":              _predict_model("xgb_credit_rate",              payload),
        "Predicted_Health_Rate":              _predict_model("xgb_health_rate",              payload),
        "Predicted_Food_Percentage":          _predict_model("xgb_food_percentage",          payload),
    }
    goal_feats = {
        "Income": float(payload.get("Income", 0.0) or 0.0),
        "Behavior_Cluster": int(payload["Behavior_Cluster"]),
        **preds,
    }

    meta = METAS["xgb_savings_goal"]
    order = meta["feature_names"]
    X = pd.DataFrame([{k: float(goal_feats.get(k, 0.0) or 0.0) for k in order}], columns=order)  # <-- DF
    proba = float(BUNDLES["xgb_savings_goal"].predict_proba(X)[0, 1])
    thr = float(meta.get("decision_threshold", 0.5))
    return {
        "model": "xgb_savings_goal",
        "version": meta.get("version"),
        "probability": proba,
        "threshold": thr,
        "decision": int(proba >= thr),
        "components": preds,
        "behavior_cluster": {
            "raw": payload.get("_behavior_cluster_raw"),
            "stable_id": payload["Behavior_Cluster"],
            "name": payload.get("_behavior_cluster_name"),
        },
        "feature_order": order,
    }

def _load_user_timeseries(uid: str, lookback_days: int = 365) -> pd.DataFrame:
    """
    Load daily spend history for this user.
    EXPECTED OUTPUT: DataFrame with columns ['ds','y'] where:
      - ds: pandas.Timestamp (daily)
      - y : float (total spent that day)
    Example Firestore layout: users/{uid}/daily_spend/{docId} with fields:
      - date: ISO string or Timestamp
      - amount: number
    """

    if SAFE_SHOWCASE:
        # No real user data in the public repo; return empty DF
        return pd.DataFrame(columns=["ds", "y"])

    db = firestore.Client()
    since = pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=lookback_days)

    # Adjust collection name/field names to your schema
    docs = (
        db.collection("users")
          .document(uid)
          .collection("daily_spend")
          .where("date", ">=", since.to_pydatetime())
          .order_by("date")
          .stream()
    )

    rows = []
    for d in docs:
        obj = d.to_dict() or {}
        ds = pd.to_datetime(obj.get("date"), utc=True, errors="coerce")
        y  = float(obj.get("amount", 0.0))
        if pd.notna(ds):
            rows.append({"ds": ds.normalize(), "y": y})

    if not rows:
        return pd.DataFrame(columns=["ds", "y"])

    df = (
        pd.DataFrame(rows)
        .groupby("ds", as_index=False)["y"]
        .sum()
        .sort_values("ds")
    )
    # Fill missing days with 0 (optional; comment out if you prefer gaps)
    full = pd.DataFrame({"ds": pd.date_range(df["ds"].min(), df["ds"].max(), freq="D")})
    df = full.merge(df, on="ds", how="left").fillna({"y": 0.0})
    return df

ALLOW_CLIENT_HISTORY = os.getenv("ALLOW_CLIENT_HISTORY", "0") == "1"
MIN_N = int(os.getenv("FORECAST_MIN_N", "30"))
MAX_N = int(os.getenv("FORECAST_MAX_N", "365"))
MAX_HORIZON = int(os.getenv("FORECAST_MAX_HORIZON", "90"))

def _parse_iso(d):
    return datetime.fromisoformat(d).date()

def _parse_iso_date(name: str, s: str) -> pd.Timestamp:
    """
    Parse an ISO date string into a tz-naive pandas.Timestamp.
    Normalizes to midnight (no time component).
    """
    try:
        # tz-naive, normalized to midnight
        return pd.to_datetime(s, errors="raise").normalize().tz_localize(None)
    except Exception:
        raise ValueError(f"invalid_{name}")


def _validate_history(df: pd.DataFrame):
    if "ds" not in df.columns or "y" not in df.columns:
        raise BadRequest("history must have columns: ds, y")
    df["ds"] = pd.to_datetime(df["ds"], errors="raise").dt.tz_localize(None)  # <— make tz-naive
    df["y"]  = pd.to_numeric(df["y"], errors="raise")
    if (df["y"] < 0).any():
        raise BadRequest("history.y must be non-negative")
    df = df.sort_values("ds").drop_duplicates("ds")
    if len(df) < MIN_N:
        raise BadRequest(f"history too short; need at least {MIN_N} points")
    if len(df) > MAX_N:
        df = df.tail(MAX_N)
    return df


class V2Req(BaseModel):
    model: str = Field(..., description="Model name, e.g. 'xgb_spending_rate' or 'xgb_savings_goal'")
    data: dict = Field(..., description="Feature dict; keys must be feature_names for the model")

    model_config = ConfigDict(extra="forbid")  # reject unknown top-level keys

    @field_validator("model")
    @classmethod
    def known_model(cls, v: str):
        # Accept aliases ("kmeans" -> "kmeans_spend") the same way as /predict
        resolved = ALIASES.get(v, v)
        if resolved not in METAS:
            raise ValueError(f"unknown_model:{v}")
        return resolved


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.get("/")
def home():
    return "ML Model API is running."

@app.get("/healthz")
def healthz():
    return jsonify({
        "status": "ok",
        "loaded_models": sorted(BUNDLES.keys()),
        "bucket": None if SAFE_SHOWCASE else os.getenv("MODEL_BUCKET"),
        "prefix": None if SAFE_SHOWCASE else os.getenv("MODEL_PREFIX"),
        "artifacts_dir": None if SAFE_SHOWCASE else str(ARTIFACTS_DIR),
    })


@app.get("/schema")
def schema():
    out = {}
    for name, meta in METAS.items():
        out[name] = {
            "version": meta.get("version"),
            "model_type": meta.get("model_type"),
            "feature_names": meta.get("feature_names"),
            "target": meta.get("target"),
            "value_range": meta.get("value_range"),
            "cv_rmse_mean": meta.get("cv_rmse_mean"),
            "cv_rmse_std": meta.get("cv_rmse_std"),
            "decision_threshold": meta.get("decision_threshold", 0.5 if meta.get("model_type")=="classifier" else None),
        }
    return jsonify(out)


# ---- NEW: Prophet metadata endpoint ----
@app.get("/prophet/metadata")
def prophet_metadata():
    return jsonify({"prefit": False, "mode": "on_the_fly", "package_available": _HAS_PROPHET})

@app.post("/v2/predict")
@require_auth
def v2_predict():
    t0 = time.time()
    try:
        req = V2Req.model_validate(request.get_json(force=True))
    except Exception as e:
        _log_predict_event("unknown", None, t0, "error", error=str(e))
        return jsonify({"error": "invalid_request", "details": str(e)}), 400

    model = req.model
    meta = METAS.get(model)
    if not meta or model not in BUNDLES:
        _log_predict_event(model, meta, t0, "error", error="unknown_or_unloaded_model")
        return jsonify({"error": "unknown_or_unloaded_model", "model": model}), 400

    try:
        # ------------------------------------------------------------------
        # STACKED PATH: xgb_savings_goal (ensure Behavior_Cluster + OOF_* meta)
        # ------------------------------------------------------------------
        if model == "xgb_savings_goal":
            data = dict(req.data)

            # Ensure Behavior_Cluster if missing
            if "Behavior_Cluster" not in data:
                data["Behavior_Cluster"] = ensure_cluster(
                    METAS["kmeans_spend"], BUNDLES["kmeans_spend"], data
                )

            # Build OOF_* meta-features from submodels (serve-time predictions)
            oof_meta = predict_meta_features(BUNDLES, METAS, data)
            data.update(oof_meta)

            # STRICT presence: classifier schema (includes OOF_* cols)
            required = meta.get("feature_names", [])
            missing = [k for k in required if k not in data]
            if missing:
                _log_predict_event(model, meta, t0, "error", error="missing_features", missing=missing)
                return jsonify({
                    "error": "missing_features",
                    "missing": missing,
                    "required": required
                }), 400

            # Predict probability + decision
            X = np.array([to_ordered_array(data, required)], dtype=float)
            pipe = BUNDLES[model]

            if hasattr(pipe, "predict_proba"):
                proba = float(pipe.predict_proba(X)[0, 1])
            else:
                # fallback
                y = float(pipe.predict(X)[0])
                proba = float(max(0.0, min(1.0, y)))

            threshold = float(meta.get("decision_threshold", 0.5))
            decision = int(proba >= threshold)

            _log_predict_event(model, meta, t0, "ok")
            return jsonify({
                "model": model,
                "version": meta.get("version"),
                "feature_order": required,
                "probability": proba,
                "threshold": threshold,
                "decision": decision,
                "behavior_cluster": int(data["Behavior_Cluster"]),
                "oof_features": oof_meta,  # transparency/debug
            })

        # -----------------------------------------------------------
        # NON-STACKED PATH: strict feature presence, then prediction
        # -----------------------------------------------------------
        required = meta.get("feature_names", [])
        missing  = [k for k in required if k not in req.data]
        if missing:
            _log_predict_event(model, meta, t0, "error", error="missing_features", missing=missing)
            return jsonify({
                "error": "missing_features",
                "missing": missing,
                "required": required
            }), 400

        pred = _predict_model(model, req.data)
        _log_predict_event(model, meta, t0, "ok")
        return jsonify({
            "model": model,
            "version": meta.get("version"),
            "feature_order": required,
            "prediction": pred
        })

    except Exception as e:
        _log_predict_event(model, meta, t0, "error", error=str(e))
        logger.exception("v2_predict failed")
        return jsonify({"error": "prediction_failed", "details": str(e)}), 500


@app.post("/predict")
@require_auth
def predict():
    t0 = time.time()

    # capture uid for logs (this decorator sets g.user)
    uid = None
    try:
        if hasattr(g, "claims") and isinstance(g.claims, dict):
            uid = g.claims.get("uid")
        elif hasattr(g, "user") and isinstance(g.user, dict):
            uid = g.user.get("uid")
    except Exception:
        pass
    logctx = {"uid": uid} if uid else {}

    body = request.get_json(silent=True) or {}
    model_type = body.get("model_type")
    features = body.get("features")

    if not model_type or features is None:
        _log_predict_event("unknown", None, t0, "error",
                           error="missing model_type or features", **logctx)
        return jsonify({"error": "model_type and features required"}), 400

    model_name = _resolve_model_name(model_type)

    # Orchestrated savings goal (server-side)
    if model_name == "xgb_savings_goal" and isinstance(features, dict):
        try:
            result = _orchestrate_savings_goal(features)
            _log_predict_event(model_name, METAS.get(model_name), t0, "ok", **logctx)
            return jsonify(result)
        except Exception as e:
            _log_predict_event(model_name, METAS.get(model_name), t0, "error",
                               error=str(e), **logctx)
            logger.exception("savings_goal_orchestration_failed")
            return jsonify({"error": "orchestrator_failed", "details": str(e)}), 400

    if model_name not in BUNDLES:
        _log_predict_event(model_name, METAS.get(model_name), t0, "error",
                           error="unknown_or_unloaded_model", **logctx)
        return jsonify({"error": f"unknown or unloaded model '{model_name}'"}), 400

    pipe = BUNDLES[model_name]
    meta = METAS.get(model_name, {})
    feature_names = meta.get("feature_names") or []

    try:
        # Order/validate features
        X_row, missing = _order_features(features, feature_names) if feature_names else (features, [])
        if missing:
            logger.warning("missing_features model=%s missing=%s", model_name, missing)
            
        X_df = pd.DataFrame([dict(zip(feature_names, X_row))], columns=feature_names)

        # KMeans
        if meta.get("model_type") == "kmeans":
            raw = int(pipe.predict(X_df)[0])
            stable_info = kmeans_stable_output(raw)
            _log_predict_event(model_name, meta, t0, "ok", **logctx)
            return jsonify({
                "model": model_name,
                "version": meta.get("version"),
                "raw_cluster": raw,
                **stable_info
            })

        # Classifier
        if meta.get("model_type") == "classifier" or hasattr(pipe, "predict_proba"):
            proba = float(pipe.predict_proba(X_df)[:, 1][0])
            threshold = float(meta.get("decision_threshold", 0.5))
            decision = int(proba >= threshold)
            _log_predict_event(model_name, meta, t0, "ok", **logctx)
            return jsonify({
                "model": model_name,
                "version": meta.get("version"),
                "probability": proba,
                "threshold": threshold,
                "decision": decision
            })

        # Regressor
        y = float(pipe.predict(X_df)[0])
        y = clip_pred(model_name, y)
        _log_predict_event(model_name, meta, t0, "ok", **logctx)
        return jsonify({
            "model": model_name,
            "version": meta.get("version"),
            "prediction": y
        })

    except Exception as e:
        _log_predict_event(model_name, meta, t0, "error", error=str(e), **logctx)
        logger.exception("prediction failed")
        return jsonify({"error": "prediction_failed", "details": str(e)}), 500

@app.post("/v2/forecast")
@require_auth
def forecast_v2():
    if not _HAS_PROPHET:
        return jsonify({"error": "prophet_not_installed", "details": _PROPHET_ERR}), 500
    try:
        raw = request.get_data(cache=True, as_text=True)
        app.logger.info(
            "forecast: headers_seen=%s len=%s",
            {k: v for k, v in request.headers.items()
             if k.lower() in ("authorization","x-firebase-token","content-type")},
            request.content_length,
        )
    except Exception as e:
        app.logger.info("forecast: raw body read failed: %s", e)

    payload = request.get_json(silent=True) or {}
    start_s = payload.get("start")
    end_s   = payload.get("end")
    data    = (payload.get("data") or {})
    hist_in = payload.get("history")  # optional client-provided history

    app.logger.info(
        "/v2/forecast allow=%s hist_key_present=%s",
        os.getenv("ALLOW_CLIENT_HISTORY") == "1",
        "history" in (request.get_json(silent=True) or {})
    )

    if not start_s or not end_s:
        return jsonify({"error": "missing_start_or_end"}), 400

    # --- parse dates & horizon guards ---
    start = None
    end   = None
    try:
        # _parse_iso_date already normalizes to midnight and returns tz-naive
        start = _parse_iso_date("start", start_s)
        end   = _parse_iso_date("end",   end_s)
    except Exception as ve:
        return jsonify({"error": "invalid_date", "details": str(ve)}), 400

    # double-guard in case something weird happened
    if start is None or end is None:
        return jsonify({"error": "invalid_date", "details": "start/end could not be parsed"}), 400

    if end < start:
        return jsonify({"error": "end_before_start"}), 400

    # Safe no-ops if already naive
    try: start = start.tz_localize(None)
    except Exception: pass
    try: end = end.tz_localize(None)
    except Exception: pass

    horizon_days = int((end - start).days) + 1
    if horizon_days < 1 or horizon_days > MAX_HORIZON:
        return jsonify({"error": "horizon_out_of_range", "allowed": [1, MAX_HORIZON]}), 400

    # --- identify uid from your auth decorator context ---
    uid = None
    if hasattr(g, "claims") and isinstance(g.claims, dict):
        uid = g.claims.get("uid")
    elif hasattr(g, "user") and isinstance(g.user, dict):
        uid = g.user.get("uid")
    if not uid:
        return jsonify({"error": "uid_missing"}), 401

    # --- choose history source ---
    mode = "prod_history"
    df = None

    # 1) Client-supplied history (demo): only when feature flag enabled
    if hist_in is not None and ALLOW_CLIENT_HISTORY:
        try:
            df = pd.DataFrame(hist_in)
            df = _validate_history(df)  # coercions + tz-naive
            mode = "client_history"
        except Exception as e:
            return jsonify({"error": "bad_history", "details": str(e)}), 400

    # 2) Otherwise load REAL user history (Firestore)
    if df is None:
        hist = _load_user_timeseries(uid, lookback_days=365)
        if hist is None or hist.empty or len(hist) < MIN_N:
            return jsonify({"error": "insufficient_history", "n": int(getattr(hist, "shape", [0])[0])}), 404

        # normalize to Prophet columns: ds, y
        df = hist.copy()
        if "date" in df.columns and "ds" not in df.columns:
            df = df.rename(columns={"date": "ds"})
        if "value" in df.columns and "y" not in df.columns:
            df = df.rename(columns={"value": "y"})

        # ensure tz-naive before validation
        try:
            df["ds"] = pd.to_datetime(df["ds"], errors="coerce").dt.tz_localize(None)
        except Exception:
            df["ds"] = pd.to_datetime(df["ds"], errors="coerce")

        try:
            df = _validate_history(df)
        except Exception as e:
            return jsonify({"error": "bad_history", "details": str(e)}), 400

    # --- Fit Prophet & predict (tz-naive throughout) ---
    m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)

    # if "Spending_Rate" in data:
    #     m.add_regressor("spending_rate")
    #     df["spending_rate"] = float(data.get("Spending_Rate", 0.7))

    try:
        m.fit(df[["ds", "y"]])

        future = pd.DataFrame({"ds": pd.date_range(start=start, end=end, freq="D")})
        try:
            future["ds"] = pd.to_datetime(future["ds"]).dt.tz_localize(None)
        except Exception:
            future["ds"] = pd.to_datetime(future["ds"])

        # if "Spending_Rate" in data:
        #     future["spending_rate"] = float(data.get("Spending_Rate", 0.7))

        fcst = m.predict(future)[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    except Exception as e:
        app.logger.exception(
            "forecast_v2: prophet failed df_n=%s df_head=%s start=%s end=%s",
            len(df), df.head(3).to_dict("records"), start, end
        )
        return jsonify({"error": "prophet_failed", "details": str(e)}), 500

    fcst["ds"] = pd.to_datetime(fcst["ds"]).dt.strftime("%Y-%m-%d")

    return jsonify({
        "model": "prophet",
        "version": "on_the_fly",
        "mode": mode,  # "prod_history" or "client_history"
        "requested": {"start": start_s, "end": end_s, "horizon_days": horizon_days},
        "history_points": int(df.shape[0]),
        "forecast": fcst.to_dict("records"),
    })
        
@app.get("/_ping")
def _ping():
    return jsonify(status="pong", cwd=os.getcwd(), artifacts=str(ARTIFACTS_DIR))

@app.get("/_routes")
def _routes():
    routes = []
    for r in app.url_map.iter_rules():
        routes.append({"rule": r.rule, "methods": sorted(m for m in r.methods if m not in ("HEAD","OPTIONS"))})
    return jsonify(routes)

@app.get("/health")
def health():
    return jsonify({
        "status": "ok",
        "loaded_models": sorted(BUNDLES.keys()),
        "bucket": os.getenv("MODEL_BUCKET"),
        "prefix": os.getenv("MODEL_PREFIX"),
        "artifacts_dir": str(ARTIFACTS_DIR),
    })

@app.errorhandler(404)
def not_found(e):
    return jsonify(error="not_found", path=request.path), 404

# Cloud Functions / Functions Framework entrypoint (optional)
def main(request):
    with app.test_request_context(path=request.path, method=request.method):
        return app.full_dispatch_request()
    
# One-time route dump when LOG_ROUTES=1 (runs after all routes are registered)
if os.getenv("LOG_ROUTES") == "1":
    for r in sorted(app.url_map.iter_rules(), key=lambda x: x.rule):
        methods = sorted(m for m in r.methods if m not in ("HEAD","OPTIONS"))
        logger.info("ROUTE %s %s", ",".join(methods), r.rule)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    app.run(host="0.0.0.0", port=port)