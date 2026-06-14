#!/usr/bin/env bash
# Build LTX-WS Videofentanyl macOS .app with PyInstaller.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ZIP=false
for arg in "$@"; do
  case "$arg" in
    --zip) ZIP=true ;;
  esac
done

APP_NAME="LTX-WS Videofentanyl.app"
APP_PATH="dist/${APP_NAME}"

echo "==> Building Web UI"
cd web
if [[ "${CI:-}" == "true" ]] && [[ -f package-lock.json ]]; then
  npm ci
else
  if [[ ! -d node_modules ]]; then
    npm install
  fi
fi
npm run build
cd "$ROOT"

if ! python3 -c "import PyInstaller" 2>/dev/null; then
  echo "==> Installing PyInstaller"
  python3 -m pip install -r requirements-build.txt
fi

echo "==> PyInstaller (onedir .app)"
python3 -m PyInstaller pyinstaller/ltx_ws.spec --noconfirm --clean

if [[ ! -d "$APP_PATH" ]]; then
  echo "Error: expected app bundle at ${APP_PATH}"
  exit 1
fi

if [[ "$ZIP" == true ]]; then
  VERSION="${RELEASE_VERSION:-}"
  if [[ -z "$VERSION" ]]; then
    VERSION="$(git describe --tags --always 2>/dev/null || echo dev)"
  fi
  # Safe filename segment (tags may contain slashes in rare cases).
  SAFE_VERSION="${VERSION//\//-}"
  ZIP_NAME="LTX-WS-Videofentanyl-${SAFE_VERSION}-macos-arm64.zip"
  ZIP_PATH="dist/${ZIP_NAME}"
  rm -f "$ZIP_PATH"
  echo "==> Zipping ${APP_NAME} → ${ZIP_NAME}"
  ditto -c -k --sequesterRsrc --keepParent "$APP_PATH" "$ZIP_PATH"
  echo "Done: ${ZIP_PATH}"
else
  echo ""
  echo "Done: ${APP_PATH}"
  echo "Logs (frozen): ~/Library/Application Support/LTX-WS/logs/ltx-ws.log"
  echo "Models:        ~/Library/Application Support/LTX-WS/models/"
fi
