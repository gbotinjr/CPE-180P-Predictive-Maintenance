"""PSI drift check. Exit code 1 if any feature PSI > 0.2. Run via cron/systemd timer."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from app.features import FEATURE_NAMES, build_features

LOG, REF, THRESH, MIN_ROWS = Path("logs/requests.jsonl"), Path("models/reference_hist.json"), 0.2, 50


def psi(edges, ref_props, values):
    edges = np.array(edges, dtype=float)
    edges[0], edges[-1] = -np.inf, np.inf  # catch out-of-range values
    cur = np.histogram(values, bins=edges)[0] / max(len(values), 1)
    e, a = np.clip(ref_props, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def main():
    rows = [json.loads(l)["features"] for l in LOG.read_text().splitlines() if l.strip()]
    if len(rows) < MIN_ROWS:
        print(f"only {len(rows)} requests logged (need {MIN_ROWS}); skipping")
        return 0
    X = build_features(pd.DataFrame(rows))
    ref, drifted = json.loads(REF.read_text()), []
    for i, f in enumerate(FEATURE_NAMES):
        v = psi(ref[f]["edges"], ref[f]["props"], X[:, i])
        flag = "DRIFT" if v > THRESH else "ok"
        print(f"{f:22s} PSI={v:6.3f} {flag}")
        if v > THRESH:
            drifted.append(f)
    if drifted:
        print("ALERT: drift in", ", ".join(drifted))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
