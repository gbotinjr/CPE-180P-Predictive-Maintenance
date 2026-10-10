"""Train, compare, export to ONNX, parity-test, and write metadata + PSI reference histograms.

Usage:  python train.py                 # needs data/ai4i2020.csv
        python train.py --synthetic     # fake data, for smoke tests only
"""
import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import pandera.pandas as pa
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from app.features import FEATURE_NAMES, RAW_NUMERIC, build_features

SEED = 42
DATA = Path("data/ai4i2020.csv")
MODEL_VERSION = "1.0.0"
RENAME = {
    "Type": "type", "Air temperature [K]": "air_temp_k", "Process temperature [K]": "process_temp_k",
    "Rotational speed [rpm]": "rotational_speed_rpm", "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min", "Machine failure": "failure",
}
SCHEMA = pa.DataFrameSchema({
    "type": pa.Column(str, pa.Check.isin(["L", "M", "H"])),
    "air_temp_k": pa.Column(float, nullable=False),
    "process_temp_k": pa.Column(float, nullable=False),
    "rotational_speed_rpm": pa.Column(float, pa.Check.gt(0)),
    "torque_nm": pa.Column(float, pa.Check.gt(0)),
    "tool_wear_min": pa.Column(float, pa.Check.ge(0)),
    "failure": pa.Column(int, pa.Check.isin([0, 1])),
})


def synthetic(n=10000):
    rng = np.random.default_rng(SEED)
    df = pd.DataFrame({
        "type": rng.choice(["L", "M", "H"], n, p=[.6, .3, .1]),
        "air_temp_k": rng.normal(300, 2, n), "process_temp_k": rng.normal(310, 1.5, n),
        "rotational_speed_rpm": rng.normal(1540, 170, n).clip(1100), "torque_nm": rng.normal(40, 10, n).clip(3),
        "tool_wear_min": rng.uniform(0, 250, n),
    })
    risk = (df.torque_nm > 55) | (df.tool_wear_min > 220) | (df.rotational_speed_rpm < 1250)
    df["failure"] = (risk & (rng.random(n) < .6)).astype(int)
    return df


def load_data(synth=False):
    if synth:
        df = synthetic()
    else:
        if not DATA.exists():
            sys.exit(f"{DATA} missing. Download AI4I 2020 from UCI and save it there (see README).")
        df = pd.read_csv(DATA, encoding="utf-8-sig").rename(columns=RENAME)  # leakage: UDI, Product ID, TWF..RNF never selected
        df = df[list(RENAME.values())]
    df = df.astype({c: float for c in RAW_NUMERIC})
    return SCHEMA.validate(df)


def split(df):
    X, y = build_features(df), df["failure"].to_numpy()
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def make_pipe(clf, scale=True):
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:  # trees do not need scaling; skipping it removes a float32/float64 mismatch
        steps.append(("scaler", StandardScaler()))
    return Pipeline([*steps, ("clf", clf)])


def candidates():
    return {
        "logreg": make_pipe(LogisticRegression(class_weight="balanced", max_iter=1000)),
        "random_forest": make_pipe(RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced_subsample",
                                                          random_state=SEED, n_jobs=-1), scale=False),
        "gradient_boosting": make_pipe(GradientBoostingClassifier(random_state=SEED), scale=False),
    }


def fit(name, pipe, X, y):
    if name == "gradient_boosting":  # no class_weight param -> sample weights
        pipe.fit(X, y, clf__sample_weight=compute_sample_weight("balanced", y))
    else:
        pipe.fit(X, y)
    return pipe


def metrics(y, proba):
    pred = (proba >= 0.5).astype(int)
    return {"accuracy": accuracy_score(y, pred), "failure_recall": recall_score(y, pred), "roc_auc": roc_auc_score(y, proba)}


def to_onnx(pipe):
    clf = pipe.named_steps["clf"]
    return convert_sklearn(pipe, initial_types=[("input", FloatTensorType([None, len(FEATURE_NAMES)]))],
                           options={id(clf): {"zipmap": False}}, target_opset={"": 17, "ai.onnx.ml": 3})


def onnx_proba(path, X):
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return sess.run(None, {"input": X.astype(np.float32)})[1][:, 1]


def main(synth):
    df = load_data(synth)
    print(f"rows={len(df)} failure_rate={df.failure.mean():.3%}")
    Xtr, Xte, ytr, yte = split(df)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    results = {}
    for name, pipe in candidates().items():
        kw = {"params": {"clf__sample_weight": compute_sample_weight("balanced", ytr)}} if name == "gradient_boosting" else {}
        cvp = cross_val_predict(pipe, Xtr, ytr, cv=cv, method="predict_proba", **kw)[:, 1]
        cvm = metrics(ytr, cvp)
        print(f"{name:18s} 5-fold CV: recall={cvm['failure_recall']:.3f} auc={cvm['roc_auc']:.3f}")
        fit(name, pipe, Xtr, ytr)
        m = metrics(yte, pipe.predict_proba(Xte)[:, 1])
        results[name] = (pipe, m)
        print(f"{name:18s} " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))
    ok = {n: r for n, r in results.items() if r[1]["accuracy"] >= 0.90}
    if not ok:
        sys.exit("No candidate reaches 90% accuracy.")
    best = max(ok, key=lambda n: (ok[n][1]["failure_recall"], ok[n][1]["roc_auc"]))
    pipe, m = ok[best]
    print("selected:", best)

    Path("models").mkdir(exist_ok=True)
    onx_path = Path("models/model.onnx")
    onx_path.write_bytes(to_onnx(pipe).SerializeToString())
    sk_p, ox_p = pipe.predict_proba(Xte)[:, 1], onnx_proba(onx_path, Xte)
    diff = np.abs(ox_p - sk_p)
    max_diff = float(diff.max())
    flips = int(((sk_p >= 0.5) != (ox_p >= 0.5)).sum())
    print(f"ONNX parity max abs diff = {max_diff:.2e} | rows > 1e-4: {(diff > 1e-4).sum()}/{len(diff)} | label flips: {flips}")
    if max_diff > 1e-4:
        sys.exit("PARITY FAILED - do not release this model.")

    ranges = {}
    for c in RAW_NUMERIC:
        lo, hi = float(df[c].min()), float(df[c].max())
        pad = 0.1 * (hi - lo)
        ranges[c] = [max(lo - pad, 0.0) if c in ("rotational_speed_rpm", "torque_nm", "tool_wear_min") else lo - pad, hi + pad]
    ref = {}
    for i, f in enumerate(FEATURE_NAMES):
        counts, edges = np.histogram(Xtr[:, i], bins=10)
        ref[f] = {"edges": edges.tolist(), "props": (counts / counts.sum()).tolist()}
    meta = {"model_version": MODEL_VERSION, "model_type": best, "training_date": str(date.today()),
            "metrics": m, "parity_max_abs_diff": max_diff, "synthetic_data": synth,
            "sha256": hashlib.sha256(onx_path.read_bytes()).hexdigest(), "ranges": ranges}
    Path("models/metadata.json").write_text(json.dumps(meta, indent=2))
    Path("models/reference_hist.json").write_text(json.dumps(ref))
    print("wrote models/model.onnx, metadata.json, reference_hist.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    main(ap.parse_args().synthetic)
