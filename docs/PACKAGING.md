# Packaging: macOS app (PyInstaller)

Build a double-clickable **LTX-WS Videofentanyl** app for Apple Silicon.

## Prerequisites

- macOS on Apple Silicon
- Python 3.11+ venv with all runtime deps (`requirements.txt` + ltx-2-mlx v0.14.9)
- Node.js 18+ (`cd web && npm install`)
- PyInstaller: `pip install pyinstaller`

## Build (local)

```bash
chmod +x scripts/build_mac_app.sh scripts/ci_install_build_deps.sh
./scripts/ci_install_build_deps.sh
./scripts/build_mac_app.sh
```

Output: `dist/LTX-WS Videofentanyl.app`

Zip for distribution:

```bash
./scripts/build_mac_app.sh --zip
# dist/LTX-WS-Videofentanyl-<version>-macos-arm64.zip
```

## GitHub Releases (CI)

Publishing a GitHub Release runs [.github/workflows/release.yml](../.github/workflows/release.yml):

1. Builds the Web UI (`npm ci` + `vite build`)
2. Installs Python deps via `scripts/ci_install_build_deps.sh`
3. Runs PyInstaller on `macos-14` (Apple Silicon)
4. Zips the `.app` and attaches `LTX-WS-Videofentanyl-<tag>-macos-arm64.zip` to the release

**To ship a build:** create a release on GitHub (tag + publish). No local build required.

Manual test without publishing: Actions → **Release** → **Run workflow**

| Input | Purpose |
|-------|---------|
| **ref** | Branch or SHA to build (e.g. `PyInstaller`). Empty = branch you pick in the UI. |
| **release_tag** | Zip filename label (e.g. `ci-test`, `v0.2.0-rc1`). |
| **attach_to_release** | Upload to an existing GitHub Release with that tag (optional). |

Produces a workflow artifact; release attachment only when publishing a release or when `attach_to_release` is enabled.

First launch opens `http://127.0.0.1:8765/` in your browser. The app runs headless (no terminal).

## Runtime paths (frozen)

| Data | Location |
|------|----------|
| Models | `~/Library/Application Support/LTX-WS/models/` |
| LoRAs | `~/Library/Application Support/LTX-WS/loras/` |
| Web outputs | `~/Library/Application Support/LTX-WS/web_outputs/` |
| Logs | `~/Library/Application Support/LTX-WS/logs/ltx-ws.log` |

Override base: `LTX_WS_DATA_DIR=/path`

## UI status indicators

Without a console, startup progress is shown in the Web UI header:

- Model download (Hugging Face snapshot progress)
- MLX / pipeline loading
- LoRA download
- Active model when ready

Subscribe API: `GET /api/system/events` (SSE), snapshot: `GET /api/system/status`

## Dev entry (non-frozen)

```bash
python app_main.py --open-browser --model auto
```

Equivalent to `python server.py --web-ui --open-browser`.

## Notes

- MLX weights are **not** bundled; first run downloads to Application Support.
- `ffmpeg` is not bundled; autoconcat requires `ffmpeg` on PATH.
- PyInstaller + MLX is fragile across versions; test on a clean machine after building.
- For CLI/MCP, continue using `python server.py` / `videofentanyl.py` from a venv.
