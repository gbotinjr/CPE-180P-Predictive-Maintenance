#!/usr/bin/env bash
# Run ON the VM from the repo root. Keeps previous image as ml-api:previous for rollback.
set -euo pipefail
git pull --ff-only
docker tag ml-api:latest ml-api:previous 2>/dev/null || true
docker build -t ml-api:latest .
sudo systemctl restart ml-api
for i in $(seq 1 15); do
  curl -fs http://localhost:8000/healthz && { echo " deploy OK"; exit 0; }
  sleep 2
done
echo "health check failed - rolling back"
if docker image inspect ml-api:previous >/dev/null 2>&1; then
  docker tag ml-api:previous ml-api:latest
  sudo systemctl restart ml-api
else
  echo "no previous image to roll back to"
fi
exit 1
