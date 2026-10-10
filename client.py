"""Test client. python client.py --url http://<VM_IP>:8000 --n 200 [--shift]
Replays real rows from the AI4I CSV so normal traffic matches the training distribution."""
import argparse, collections, csv, random, statistics as st, time
import numpy as np
import requests


def load_rows(path):
    with open(path, encoding="utf-8-sig") as f:
        return [{"type": r["Type"], "air_temp_k": float(r["Air temperature [K]"]),
                 "process_temp_k": float(r["Process temperature [K]"]),
                 "rotational_speed_rpm": float(r["Rotational speed [rpm]"]),
                 "torque_nm": float(r["Torque [Nm]"]), "tool_wear_min": float(r["Tool wear [min]"])}
                for r in csv.DictReader(f)]


def sample(rows, shift):
    r = dict(random.choice(rows))
    if shift:  # simulate a worn, overloaded machine
        r["torque_nm"] = min(r["torque_nm"] * 1.35, 80.0)
        r["tool_wear_min"] = random.uniform(180, 250)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--data", default="data/ai4i2020.csv")
    ap.add_argument("--shift", action="store_true", help="send drifted data (high torque + wear)")
    a = ap.parse_args()
    rows, rtt, srv, errs = load_rows(a.data), [], [], collections.Counter()
    for _ in range(a.n):
        t0 = time.perf_counter()
        r = requests.post(f"{a.url}/v1/predict", json=sample(rows, a.shift), timeout=5)
        rtt.append((time.perf_counter() - t0) * 1000)
        if r.ok:
            srv.append(r.json()["latency_ms"])
        else:
            errs[r.status_code] += 1
    p95 = lambda xs: float(np.percentile(xs, 95))
    print(f"ok={len(srv)} errors={dict(errs) or 0}")
    print(f"server latency  median={st.median(srv):.1f} ms  p95={p95(srv):.1f} ms  (SLA <= 100)")
    print(f"network RTT     median={st.median(rtt):.1f} ms  p95={p95(rtt):.1f} ms  (reported only)")


if __name__ == "__main__":
    main()
