"""Single source of truth for feature engineering (imported by training AND the API)."""
import numpy as np
import pandas as pd

TYPES = ["L", "M", "H"]
RAW_NUMERIC = ["air_temp_k", "process_temp_k", "rotational_speed_rpm", "torque_nm", "tool_wear_min"]
FEATURE_NAMES = ["type_L", "type_M", "type_H", *RAW_NUMERIC, "temp_diff_k", "power_w"]
N_FEATURES = len(FEATURE_NAMES)  # 10


def build_features(df: pd.DataFrame) -> np.ndarray:
    """df needs columns: type + RAW_NUMERIC. Returns float32 array (n, 10)."""
    t = df["type"].astype(str)
    out = pd.DataFrame({f"type_{k}": (t == k).astype("float32") for k in TYPES})
    for c in RAW_NUMERIC:
        out[c] = df[c].astype("float32")
    out["temp_diff_k"] = df["process_temp_k"] - df["air_temp_k"]
    out["power_w"] = df["torque_nm"] * df["rotational_speed_rpm"] * 2 * np.pi / 60
    return out[FEATURE_NAMES].to_numpy(dtype=np.float32)
