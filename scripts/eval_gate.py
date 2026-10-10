"""CI evaluation gate: re-evaluate the committed ONNX model on the held-out split."""
import sys

from train import load_data, metrics, onnx_proba, split

df = load_data()
_, Xte, _, yte = split(df)
m = metrics(yte, onnx_proba("models/model.onnx", Xte))
print(m)
fail = [k for k, t in (("accuracy", .90), ("failure_recall", .70)) if m[k] < t]
sys.exit(f"GATE FAILED: {fail}" if fail else 0)
