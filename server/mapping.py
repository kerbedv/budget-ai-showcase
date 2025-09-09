from typing import Dict, List

def to_ordered_array(payload: Dict[str, object], feature_order: List[str]) -> list[float]:
    """
    Build a strictly-ordered numeric array from `payload` according to `feature_order`.

    Errors:
      - KeyError("missing_features::<comma-separated-keys>") when any required key is absent
      - ValueError("invalid_type::<key>") when a value cannot be coerced to float
    """
    missing = [k for k in feature_order if k not in payload]
    if missing:
        # Caller should catch this and return a 400 with {"error":"missing_features", ...}
        raise KeyError(f"missing_features::{','.join(missing)}")

    ordered: list[float] = []
    for k in feature_order:
        v = payload[k]
        try:
            ordered.append(float(v))
        except Exception:
            # Caller can map this to 400 with a helpful message
            raise ValueError(f"invalid_type::{k}")
    return ordered
