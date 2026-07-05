#!/usr/bin/env bash
# Pull latest, install Python deps (includes mediapipe), rebuild web UI.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
git pull
pip install -r requirements.txt
(
  cd web
  npm run build
)
echo "Done. Restart server.py if it is running."
