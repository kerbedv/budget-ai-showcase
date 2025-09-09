from __future__ import annotations
import os
from pathlib import Path
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

from bundle_utils import save_bundle
from common import version, data_path

# ---- Inputs (env-overridable) ----
CSV_PATH = data_path("daily_synthetic_spending_data.csv")
DS_COL   = os.getenv("PROPHET_DS", "ds")
Y_COL    = os.getenv("PROPHET_Y",  "y")

# Prophet core config (added: growth, changepoint_prior_scale, country holidays, regressors)
GROWTH = os.getenv("PROPHET_GROWTH", "linear")  # "linear" | "logistic"
CHANGEPOINT_PRIOR_SCALE = float(os.getenv("PROPHET_CHANGEPOINT_PRIOR_SCALE", "0.05"))
HOLIDAYS_COUNTRY = os.getenv("PROPHET_HOLIDAYS_COUNTRY", "").strip()  # e.g., "US"
REGRESSORS = [s.strip() for s in os.getenv("PROPHET_REGRESSORS", "").split(",") if s.strip()]

# Seasonality / holidays toggles
SEASONALITY_MODE  = os.getenv("PROPHET_SEASONALITY_MODE", "additive")
SEASONALITY_PRIOR = float(os.getenv("PROPHET_SEASONALITY_PRIOR", "3.0"))
HOLIDAYS_PRIOR    = float(os.getenv("PROPHET_HOLIDAYS_PRIOR", "3.0"))
ADD_BIWEEKLY      = os.getenv("PROPHET_ADD_BIWEEKLY", "1") == "1"
BIWEEKLY_FOURIER  = int(os.getenv("PROPHET_BIWEEKLY_FOURIER", "3"))
ENABLE_HOLIDAYS   = os.getenv("PROPHET_ENABLE_HOLIDAYS", "1") == "1"

# CV settings
CV_INITIAL = os.getenv("PROPHET_CV_INITIAL", "400 days")
CV_PERIOD  = os.getenv("PROPHET_CV_PERIOD",  "30 days")
CV_HORIZON = os.getenv("PROPHET_CV_HORIZON", "60 days")

# Optional custom holidays (used if ENABLE_HOLIDAYS)
HOLIDAYS = [
    "2023-01-01", "2023-02-14", "2023-07-04", "2023-11-24", "2023-12-25",
]

def _infer_y(df: pd.DataFrame, prefer: str) -> pd.DataFrame:
    if "ds" not in df.columns:
        raise ValueError("Prophet requires a 'ds' column (date).")
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    if df["ds"].isna().any():
        raise ValueError("Some 'ds' values could not be parsed as dates.")
    if prefer in df.columns and prefer != "y":
        df = df.rename(columns={prefer: "y"})
        return df
    for c in ["y", "total_spending", "Total_Spending", "spending", "Spending", "amount", "value"]:
        if c in df.columns:
            return df.rename(columns={c: "y"}) if c != "y" else df
    raise ValueError(f"Could not find target column for Prophet. Tried: '{prefer}', common aliases.")

def _holidays_df() -> pd.DataFrame | None:
    if not ENABLE_HOLIDAYS:
        return None
    return pd.DataFrame({
        "holiday": "custom_holiday",
        "ds": pd.to_datetime(HOLIDAYS),
        "lower_window": -1,
        "upper_window": 1,
    })

def main():
    ver = version()

    # 1) Load/normalize data
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()
    if DS_COL != "ds" and DS_COL in df.columns:
        df = df.rename(columns={DS_COL: "ds"})
    df = _infer_y(df, Y_COL)
    df = df.groupby("ds", as_index=False)["y"].sum()

    # Ensure any requested regressors exist (fill 0 if missing)
    for r in REGRESSORS:
        if r not in df.columns:
            df[r] = 0.0

    # 2) Build Prophet with explicit core config
    model = Prophet(
        growth=GROWTH,
        changepoint_prior_scale=CHANGEPOINT_PRIOR_SCALE,
        holidays=_holidays_df(),                               # custom holidays (optional)
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_prior_scale=SEASONALITY_PRIOR,
        holidays_prior_scale=HOLIDAYS_PRIOR,
        seasonality_mode=SEASONALITY_MODE,
    )

    # Country holidays (optional)
    if HOLIDAYS_COUNTRY:
        model.add_country_holidays(country_name=HOLIDAYS_COUNTRY)

    # Custom biweekly seasonality (optional)
    if ADD_BIWEEKLY:
        model.add_seasonality(name="biweekly", period=14, fourier_order=BIWEEKLY_FOURIER)

    # Extra regressors (optional)
    for r in REGRESSORS:
        model.add_regressor(r)

    # 3) Fit
    fit_cols = ["ds", "y"] + REGRESSORS
    model.fit(df[fit_cols])

    # 4) Cross-validation (best-effort)
    try:
        df_cv = cross_validation(model, initial=CV_INITIAL, period=CV_PERIOD, horizon=CV_HORIZON)
        df_p  = performance_metrics(df_cv)
        metrics = {
            "initial": CV_INITIAL, "period": CV_PERIOD, "horizon": CV_HORIZON,
            "rmse_mean": float(df_p["rmse"].mean()),
            "mae_mean":  float(df_p["mae"].mean()),
            "mape_mean": float(df_p["mape"].mean()),
            "rows": int(len(df_cv)),
        }
    except Exception as e:
        metrics = {"warning": f"cv_failed: {e.__class__.__name__}: {e}"}

    # 5) Save bundle with transparent config in metadata
    extra_meta = {
        "model_type": "prophet",
        "target": "y",
        "train_range": {
            "start": df["ds"].min().strftime("%Y-%m-%d"),
            "end":   df["ds"].max().strftime("%Y-%m-%d"),
            "n_obs": int(len(df)),
        },
        "prophet_config": {
            "growth": GROWTH,
            "seasonality_mode": SEASONALITY_MODE,
            "changepoint_prior_scale": CHANGEPOINT_PRIOR_SCALE,
            "weekly_seasonality": True,
            "yearly_seasonality": True,
            "seasonality_prior_scale": SEASONALITY_PRIOR,
            "holidays_prior_scale": HOLIDAYS_PRIOR,
            "holidays_country": HOLIDAYS_COUNTRY or None,
            "custom_holidays": HOLIDAYS if ENABLE_HOLIDAYS else [],
            "custom_biweekly": ({"period": 14, "fourier_order": BIWEEKLY_FOURIER} if ADD_BIWEEKLY else None),
            "regressors": REGRESSORS,   # names only; values are in the training data
            "freq": "D",
        },
        "cv_summary": metrics,
    }
    # Keep feature_names=["ds"] for schema compatibility; regressors are recorded in config
    save_bundle("prophet", ver, model, ["ds"], extra_meta)
    print(f"✅ Saved Prophet bundle v{ver}")

if __name__ == "__main__":
    main()
