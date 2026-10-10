# Predictive Maintenance Classifier (CPE180P)
## Local (WSL)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-train.txt
# Download AI4I 2020 CSV from UCI -> data/ai4i2020.csv
python train.py                       # or: python train.py --synthetic (smoke test only)
pytest -q
uvicorn app.main:app --port 8000      # then: python client.py --n 200
```
## VM (first time)
Clone repo to ~/pdm-mlops, run `deploy/setup_vm.sh`, put runtime settings in /etc/ml-api.env, then `./deploy/deploy.sh`.
## GitHub secrets: VM_SSH_KEY, VM_HOST, VM_USER
## Drift demo: `python client.py --shift --n 300` then `python -m scripts.psi_monitor`
