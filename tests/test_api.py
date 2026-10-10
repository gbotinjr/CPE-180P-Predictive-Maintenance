import os
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

GOOD = {"type": "M", "air_temp_k": 300, "process_temp_k": 310, "rotational_speed_rpm": 1500, "torque_nm": 40, "tool_wear_min": 100}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    d = tmp_path_factory.mktemp("m")
    if not os.path.exists("models/model.onnx"):
        subprocess.run([sys.executable, "train.py", "--synthetic"], check=True)
    os.environ["LOG_FILE"] = str(d / "req.jsonl")
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/healthz").status_code == 200


def test_predict(client):
    r = client.post("/v1/predict", json=GOOD)
    assert r.status_code == 200
    assert 0 <= r.json()["failure_probability"] <= 1


@pytest.mark.parametrize("patch", [{"type": "X"}, {"torque_nm": -5}, {"rotational_speed_rpm": 99999}])
def test_rejects_bad_input(client, patch):
    assert client.post("/v1/predict", json={**GOOD, **patch}).status_code == 422


def test_rejects_missing_field(client):
    bad = {k: v for k, v in GOOD.items() if k != "torque_nm"}
    assert client.post("/v1/predict", json=bad).status_code == 422


def test_model_info(client):
    assert "sha256" in client.get("/v1/model-info").json()
