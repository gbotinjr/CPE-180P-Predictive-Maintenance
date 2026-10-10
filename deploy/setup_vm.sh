#!/usr/bin/env bash
# One-time VM hardening (run as a sudo user AFTER your key login works).
set -euo pipefail
sudo sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/; s/^#\?PermitRootLogin.*/PermitRootLogin no/; s/^#\?PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config
sudo systemctl restart ssh
sudo apt-get update && sudo apt-get install -y docker.io fail2ban ufw git curl
sudo ufw default deny incoming && sudo ufw allow 22/tcp && sudo ufw allow 8000/tcp && sudo ufw --force enable
sudo mkdir -p /var/log/ml-api && sudo chown 1000:1000 /var/log/ml-api
sudo cp deploy/ml-api.service /etc/systemd/system/ && sudo touch /etc/ml-api.env && sudo chmod 600 /etc/ml-api.env
sudo systemctl daemon-reload && sudo systemctl enable ml-api
