# Cloud-Deployed Predictive Maintenance Classifier

CPE180P Cloud AI & MLOps, Group Project (Mapúa University)

A scikit-learn classifier trained on the UCI AI4I 2020 Predictive Maintenance dataset predicts machine failure from five sensor readings plus the product type. The model is exported to ONNX, served by a FastAPI app inside Docker, supervised by systemd, and monitored with JSON logs and a PSI drift check.

## Quick start: try it on your own device

You do not need access to our server or anyone else's computer. Everything runs locally from this repository, which already includes the trained model and the dataset. The deployed service runs on the team's own machine and is not reachable from outside, so each person tests their own copy.

**0. Get access (once).** The repository is private. The owner adds you under **Settings, Collaborators**. Accept the invitation from your email or GitHub notifications, and sign in with the same GitHub account that was invited. A 404 on the repository page means the invite has not been accepted or you are signed in to a different account.

**1. Download the project.** On the repository page, click **Code, Download ZIP**, then unzip it. The folder is usually named `CPE-180P-Predictive-Maintenance-main`. Or clone it: `git clone git@github.com:gbotinjr/CPE-180P-Predictive-Maintenance.git`.

**2. Install Python 3.11 or newer.** Check with `python --version` (or `python3 --version`). On Windows, tick **Add Python to PATH** during installation.

**3. Create an environment and install dependencies.** Open a terminal in the project folder.

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-train.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` and try again.

WSL, Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-train.txt
```

**4. Run the tests.**

```bash
pytest -q
```

Expected: `8 passed`.

**5. Start the API.** Leave this terminal running.

```bash
uvicorn app.main:app --port 8000
```

Wait for `Application startup complete`.

**6. Send requests.** Open a second terminal in the same folder, activate the environment again (step 3), then:

```bash
python client.py --url http://localhost:8000 --n 200
curl http://localhost:8000/healthz
curl http://localhost:8000/v1/model-info
```

Expected: `ok=200 errors=0` with the server p95 latency well under 100 ms. On Windows PowerShell use `curl.exe` instead of `curl`.

**7. See the drift alert.** Send normal traffic, check the monitor, then send shifted traffic and check again. Wait about a minute between the two client runs because of the rate limit.

```bash
python client.py --url http://localhost:8000 --n 250
python -m scripts.psi_monitor            # expect every feature "ok"
python client.py --url http://localhost:8000 --shift --n 250
python -m scripts.psi_monitor            # expect DRIFT on torque_nm, tool_wear_min, power_w and an ALERT line
```

To restart the comparison from a clean log, stop the server, delete `logs/requests.jsonl`, and start the server again.

**8. Try a bad request.** Torque cannot be negative, so the API should reject this with HTTP 422. A missing field or a `type` other than L, M or H is also rejected.

```bash
curl -X POST http://localhost:8000/v1/predict -H "Content-Type: application/json" \
  -d '{"type":"M","air_temp_k":300,"process_temp_k":310,"rotational_speed_rpm":1500,"torque_nm":-5,"tool_wear_min":100}'
```

On Windows PowerShell, use `curl.exe` and escape the quotes: `-d "{\"type\":\"M\",...}"`.

**9. Optional: retrain the model.** `python train.py` prints metrics for logistic regression, random forest and gradient boosting, selects a model, exports it to ONNX, and fails if the ONNX output does not match scikit-learn. If you change anything in `models/`, do it on a branch, not on `main`.

**10. Optional: run it in Docker.** `docker build -t ml-api .` then `docker run --rm -p 8000:8000 ml-api`, and repeat step 6.

| Quick start problem | Fix |
|---|---|
| `python` not found | Install Python and tick Add to PATH, or use `python3` |
| `Address already in use` on port 8000 | Stop the other server, or use `--port 8001` and the same port in `client.py --url` |
| `ModuleNotFoundError: No module named 'app'` | Run commands from the project root; use `python -m scripts.psi_monitor` |
| HTTP 429 from the API | The rate limit is 300 requests per minute per IP; wait a minute |
| `psi_monitor` finds no log | Send some requests first; the log is created when the API starts |

## 1. Architecture

```
Windows 11 + WSL2 Ubuntu
  user "gary"   -> training, tests, client.py, git push
  user "sakura" -> the "VM": SSH (key only), Docker, systemd service ml-api
                   Docker container: FastAPI + ONNX Runtime on port 8000
GitHub: CI (lint, tests, secret scan, evaluation gate)
```

- The "VM" is the `sakura` user on the team's WSL Ubuntu, reached with `ssh sakura@localhost` using a key.
- GitHub Actions cannot reach `localhost`, so CI runs the `test` job only. Deployment is manual (section 4). See section 9 for moving to a real cloud server.

| Route | Purpose |
|---|---|
| `POST /v1/predict` | Validate input, build features, run ONNX, return label, probability, model version, latency |
| `GET /healthz` | 200 only if the ONNX session loaded |
| `GET /v1/model-info` | Model version, training date, metrics, SHA-256 of `model.onnx` |

## 2. Repository layout

```
app/features.py    shared feature engineering (training AND API import this)
app/main.py        FastAPI service (validation, rate limit, size cap, JSON logs)
train.py           compare models, export ONNX, parity test, write metadata + reference histograms
client.py          replays real dataset rows; reports server latency and network RTT
scripts/           psi_monitor.py (drift), eval_gate.py (CI gate)
deploy/            ml-api.service, setup_vm.sh, deploy.sh
models/            model.onnx, metadata.json, reference_hist.json (committed)
data/              ai4i2020.csv (public dataset)
tests/             pytest suite
```

## 3. Local setup and training (user `gary`)

```bash
python3 -m venv .cpe180p && source .cpe180p/bin/activate
pip install -r requirements-train.txt
# data/ai4i2020.csv must exist (UCI AI4I 2020)
python train.py          # prints metrics, selects a model, writes models/*
pytest -q                # expect 8 passed
ruff check .
python -m scripts.eval_gate
```

`train.py` fails the run if the ONNX probabilities differ from scikit-learn by more than 1e-4. Always run scripts as modules (`python -m scripts.eval_gate`) so imports resolve.

Commit the three files in `models/` after retraining; the Docker image and CI both use the committed copies.

## 4. Deploy (user `sakura`)

First-time setup, once:

```bash
ssh -i ~/.ssh/id_rsa_wsl sakura@localhost
git clone git@github.com:gbotinjr/CPE-180P-Predictive-Maintenance.git ~/pdm-mlops   # read-only deploy key
cd ~/pdm-mlops && ./deploy/setup_vm.sh     # key-only SSH, ufw 22/8000, docker, fail2ban, systemd unit
sudo usermod -aG docker $USER              # then log out and back in
```

Every release:

```bash
cd ~/pdm-mlops && ./deploy/deploy.sh
```

`deploy.sh` pulls the repo, tags the running image as `ml-api:previous`, builds `ml-api:latest`, restarts the service, and polls `/healthz`. If the health check fails it restores `ml-api:previous` (when one exists) and exits non-zero.

Verify from `gary`:

```bash
curl http://localhost:8000/healthz
python client.py --url http://localhost:8000 --n 200
```

## 5. Operations

| Task | Command (as `sakura`) |
|---|---|
| Service status | `sudo systemctl status ml-api --no-pager` |
| Restart | `sudo systemctl restart ml-api` |
| Container logs | `docker logs ml-api` or `sudo journalctl -u ml-api -n 50` |
| Request log (JSON lines) | `/var/log/ml-api/requests.jsonl` (rotated at 5 MB, 5 backups) |
| Runtime settings | `/etc/ml-api.env` (chmod 600), e.g. `RATE_LIMIT=300` |

Do not delete the request log while the service runs; truncate it instead: `sudo truncate -s 0 /var/log/ml-api/requests.jsonl`.

## 6. Rollback

Automatic: `deploy.sh` rolls back if `/healthz` fails after a deploy.

Manual:

```bash
docker tag ml-api:previous ml-api:latest
sudo systemctl restart ml-api
curl http://localhost:8000/healthz
```

Then revert the bad commit (`git revert <sha> && git push`) and redeploy so the server checkout matches `main`. Note that `ml-api:previous` is only the image from the last deploy; deploy a known-good version before relying on it.

## 7. Drift monitoring and retraining

The PSI monitor compares logged request features with training histograms; PSI above 0.2 flags drift.

```bash
# as sakura, one-time
cd ~/pdm-mlops && ln -sf /var/log/ml-api/requests.jsonl logs/requests.jsonl
python3 -m venv .venv && . .venv/bin/activate && pip install numpy pandas
# each check
python -m scripts.psi_monitor        # exit code 1 and "ALERT" if any feature drifted
```

Demo: `python client.py --url http://localhost:8000 --shift --n 250` (from `gary`) sends high torque and tool wear; expect `DRIFT` on `torque_nm`, `tool_wear_min`, `power_w`.

Retraining path: confirm drift is real, collect labelled data, rerun `train.py`, check the parity and metrics, bump `MODEL_VERSION` in `train.py`, commit `models/`, deploy.

## 8. Security and secrets

- No secrets in Git. `.gitignore` excludes keys, `.env` and logs; CI runs gitleaks.
- Separate keys: `id_rsa_wsl` (log in to `sakura`), `github_key` (push from `gary`), `repo_deploy_key` (read-only, used by `sakura` to pull).
- SSH: `PasswordAuthentication no`, `PermitRootLogin no`, key auth only; fail2ban enabled.
- Firewall: ufw allows only TCP 22 and 8000. Docker publishes port 8000 through its own rules, which bypass ufw, so do not claim ufw alone enforces it. On a real cloud server the provider packet filter is the enforcing control.
- API protections: Pydantic validation with training-range checks, 4 KB request cap, per-IP rate limit.

Key rotation after a leak: remove the key from `~/.ssh/authorized_keys` (SSH) or from GitHub Settings (GitHub/deploy keys), generate a new pair, install the new public key, and update any GitHub Actions secret that used it.

## 9. Moving to a real cloud server

The same scripts apply: create an Ubuntu VM, install your public key, attach a provider packet filter allowing only TCP 22 and 8000, run `deploy/setup_vm.sh`, then `deploy/deploy.sh`. To enable CI deployment, add `VM_HOST`, `VM_USER` and `VM_SSH_KEY` as GitHub Actions secrets and remove the three-line "secrets not set" guard from `.github/workflows/ci.yml`.

## 10. Measured results

| Target | Result |
|---|---|
| Test accuracy >= 90% | 0.959 |
| Failure recall >= 0.70 | 0.912 |
| ROC-AUC >= 0.90 | 0.974 |
| Server p95 latency <= 100 ms | about 9 to 12 ms (200 to 300 requests) |
| Recovery within 30 s | 7 s (`docker kill`, systemd restart) |
| ONNX parity < 1e-4 | max diff 9.3e-08, 0 label flips |
| Password SSH rejected | `Permission denied (publickey)` |
| Only 22 and 8000 open | ufw rules verified; nmap result: *(add after running `nmap -Pn -p- localhost`)* |

## 11. Troubleshooting (problems we actually hit)

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'app'` in pytest or scripts | Run `python -m scripts.<name>`; pytest uses `pythonpath = ["."]` in `pyproject.toml` |
| ONNX parity failed (about 7e-3) | Scaler plus trees under float32; tree models are built without the scaler |
| Docker build: `Temporary failure in name resolution` | WSL DNS address is unreachable from containers. Set `{"dns": ["8.8.8.8", "1.1.1.1"]}` in `/etc/docker/daemon.json` and restart Docker |
| Container restarts; `No module named 'pandas'` | Runtime `requirements.txt` must include everything `app/` imports |
| HTTP 429 from the API | Rate limit (300 per minute per IP); wait, or raise `RATE_LIMIT` in `/etc/ml-api.env` |
| `curl` connection reset or refused | Container not running yet; check `journalctl -u ml-api` |
| PSI monitor finds no log | The log file was deleted while the service held it open; truncate instead and restart the service |

## 12. Team

| Name | Student ID | Role |
|---|---|---|
| Gary Botin Jr | | Lead MLOps Engineer |
| Carl Erick Esmeria | | Data & Training Lead |
| Eruel Joseph Dizon | | Backend API Developer |

Dataset reference: S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications," 3rd International Conference on Artificial Intelligence for Industries (AI4I), 2020.
