#!/bin/bash
# Start LTX-WS.command — double-click on Mac to set up and run the Web UI.
#
# What this does (same as a manual install + python server.py):
#   1. Creates .venv if missing
#   2. Installs Python deps + MLX packages when needed
#   3. Builds the Web UI (web/dist) when missing
#   4. Clears any leftover LTX-WS server processes
#   5. Starts server.py and restarts it if it crashes
#   6. Opens the browser to the UI
#
# Live logs stay in this Terminal window (and are also saved to logs/).
# Ctrl+C asks before stopping. Closing the window kills the server immediately.
#
# Tip: In Terminal → Settings → Profiles → Shell, set “Ask before closing”
# to Always — macOS will then confirm before the window can close.

set -u

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT" || exit 1

PORT="${LTX_WS_PORT:-8765}"
MODEL="${LTX_WS_MODEL:-auto}"
UI_URL="http://127.0.0.1:${PORT}/"
LOG_DIR="$ROOT/logs"
LOG_FILE="$LOG_DIR/ltx-ws-launcher.log"
PID_FILE="$LOG_DIR/ltx-ws-server.pid"
MARKER="$ROOT/.venv/.ltx-ws-deps-ok"
MAX_BACKOFF=60
PYTHON_MIN_MINOR=11

# Force live Python logs in Terminal (pipes / redirects would otherwise buffer).
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1

SERVER_PID=""
CLEANED=0
USER_STOP=0
KEEP_OPEN_DONE=0

mkdir -p "$LOG_DIR"

say() {
  printf '%s\n' "$*"
  printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >>"$LOG_FILE"
}

run_logged() {
  "$@" 2>&1 | tee -a "$LOG_FILE"
  return "${PIPESTATUS[0]}"
}

keep_open() {
  [[ "$KEEP_OPEN_DONE" == "1" ]] && return 0
  KEEP_OPEN_DONE=1
  echo ""
  echo "────────────────────────────────────────────────────────"
  echo "  Logs above are still in this window — scroll up to copy."
  echo "  Saved copy: $LOG_FILE"
  echo "  Press Enter to close this window…"
  echo "────────────────────────────────────────────────────────"
  # stdin may already be gone if the window was closed; don't hang forever.
  if [[ -t 0 ]]; then
    read -r _ || true
  else
    sleep 2
  fi
}

have() {
  command -v "$1" >/dev/null 2>&1
}

# ── process cleanup ───────────────────────────────────────────────────

_kill_pid_tree() {
  local pid="$1"
  local sig="${2:-TERM}"
  local child
  [[ -z "$pid" ]] && return 0
  kill -0 "$pid" 2>/dev/null || return 0
  # Children first (worker threads / subprocesses).
  while read -r child; do
    [[ -n "$child" ]] || continue
    _kill_pid_tree "$child" "$sig"
  done < <(pgrep -P "$pid" 2>/dev/null || true)
  kill "-${sig}" "$pid" 2>/dev/null || true
}

stop_tracked_server() {
  local pid="${SERVER_PID:-}"
  if [[ -z "$pid" && -f "$PID_FILE" ]]; then
    pid="$(tr -d '[:space:]' <"$PID_FILE" 2>/dev/null || true)"
  fi
  if [[ -n "${pid:-}" ]]; then
    say "Stopping server (pid $pid) and child processes…"
    _kill_pid_tree "$pid" TERM
    sleep 1
    _kill_pid_tree "$pid" KILL
  fi
  SERVER_PID=""
  rm -f "$PID_FILE"
}

# PIDs listening on our UI/API port.
_pids_on_port() {
  if have lsof; then
    lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null || true
  fi
}

# PIDs clearly belonging to this repo's server.py
_pids_for_this_server() {
  local pid cmd
  # Match absolute path to this checkout's server.py
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
    if [[ "$cmd" == *"$ROOT/server.py"* ]] || [[ "$cmd" == *" server.py"* && "$cmd" == *"python"* ]]; then
      # Prefer path match; for bare "server.py" require cwd == ROOT when possible.
      if [[ "$cmd" == *"$ROOT/server.py"* ]]; then
        echo "$pid"
        continue
      fi
      if have lsof; then
        if lsof -a -p "$pid" -d cwd 2>/dev/null | grep -F "$ROOT" >/dev/null 2>&1; then
          echo "$pid"
        fi
      fi
    fi
  done < <(pgrep -f "server\.py" 2>/dev/null || true)
}

kill_dangling_ltx_ws() {
  local pid cmd killed=0

  # 1) Stale pid file
  if [[ -f "$PID_FILE" ]]; then
    pid="$(tr -d '[:space:]' <"$PID_FILE" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      say "Found leftover pid-file server (pid $pid) — killing…"
      _kill_pid_tree "$pid" TERM
      sleep 1
      _kill_pid_tree "$pid" KILL
      killed=1
    fi
    rm -f "$PID_FILE"
  fi

  # 2) Anything running this repo's server.py
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    # Don't kill ourselves
    [[ "$pid" == "$$" ]] && continue
    cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
    say "Found dangling LTX-WS process (pid $pid) — killing…"
    say "  $cmd"
    _kill_pid_tree "$pid" TERM
    sleep 1
    _kill_pid_tree "$pid" KILL
    killed=1
  done < <(_pids_for_this_server)

  # 3) Listeners on our port that look like Python / server.py
  while read -r pid; do
    [[ -n "$pid" ]] || continue
    cmd="$(ps -p "$pid" -o args= 2>/dev/null || true)"
    if [[ "$cmd" == *"server.py"* ]] || [[ "$cmd" == *"python"* ]]; then
      say "Port $PORT still held by pid $pid — killing…"
      say "  $cmd"
      _kill_pid_tree "$pid" TERM
      sleep 1
      _kill_pid_tree "$pid" KILL
      killed=1
    fi
  done < <(_pids_on_port)

  if [[ "$killed" == "1" ]]; then
    sleep 1
    say "✓ Cleared leftover processes"
  fi
}

cleanup_all() {
  [[ "$CLEANED" == "1" ]] && return 0
  CLEANED=1
  stop_tracked_server
  kill_dangling_ltx_ws
}

confirm_stop() {
  local btn
  # Prefer a macOS dialog so it works even when focus is on the browser.
  btn="$(
    osascript <<'EOF' 2>/dev/null
try
  set msg to "Stop LTX-WS?" & return & return & "This will shut down the server and cancel any running or queued generation jobs."
  set theResult to button returned of (display dialog msg buttons {"Keep running", "Stop"} default button "Stop" with icon caution with title "LTX-WS")
  return theResult
on error
  return "Keep running"
end try
EOF
  )" || true

  if [[ "$btn" == "Stop" ]]; then
    return 0
  fi
  if [[ "$btn" == "Keep running" ]]; then
    return 1
  fi

  # Fallback: Terminal prompt
  if [[ -t 0 ]]; then
    printf 'Stop LTX-WS and cancel any running jobs? [y/N] '
    local ans=""
    read -r ans || true
    [[ "$ans" == "y" || "$ans" == "Y" || "$ans" == "yes" ]] && return 0
    return 1
  fi

  # No UI and no TTY (window dying) — stop.
  return 0
}

on_int() {
  # Ctrl+C — ask first; if they keep running, resume waiting on the server.
  trap - INT
  if confirm_stop; then
    USER_STOP=1
    say "Stopping at your request…"
    cleanup_all
    exit 130
  fi
  say "Keeping LTX-WS running. Press Ctrl+C again if you want to stop."
  trap on_int INT
}

on_hup_term() {
  # Window close / kill — no time to ask; tear everything down.
  USER_STOP=1
  say "Terminal closing — shutting down server and jobs…"
  cleanup_all
  exit 143
}

on_exit() {
  cleanup_all
  keep_open
}

trap on_exit EXIT
trap on_int INT
trap on_hup_term HUP TERM

banner() {
  clear 2>/dev/null || true
  say "╔══════════════════════════════════════════════════════╗"
  say "║              LTX-WS — easy Mac launcher              ║"
  say "╚══════════════════════════════════════════════════════╝"
  say ""
  say "Folder: $ROOT"
  say "UI:     $UI_URL"
  say "Log:    $LOG_FILE"
  say ""
  say "This Terminal window shows live logs while LTX-WS runs."
  say "If something breaks, scroll up and copy the text, or send:"
  say "  $LOG_FILE"
  say ""
  say "Stop: Ctrl+C (asks first) · Closing this window kills the server."
  say ""
}

die() {
  say "ERROR: $*"
  say ""
  say "── last log lines ──"
  if [[ -f "$LOG_FILE" ]]; then
    tail -n 40 "$LOG_FILE" 2>/dev/null || true
  fi
  exit 1
}

find_python() {
  local cand
  for cand in python3.12 python3.13 python3.11 python3; do
    if have "$cand"; then
      if "$cand" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, ${PYTHON_MIN_MINOR}) else 1)" 2>/dev/null; then
        echo "$cand"
        return 0
      fi
    fi
  done
  return 1
}

mlx_tag() {
  local tag
  tag="$(
    sed -n 's/^LTX2_MLX_GIT_TAG *= *"\([^"]*\)".*/\1/p' "$ROOT/ltx_mlx_backend.py" 2>/dev/null | head -1
  )"
  if [[ -n "${tag:-}" ]]; then
    echo "$tag"
  else
    echo "v0.14.19"
  fi
}

ensure_venv() {
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    say "✓ Python environment ready (.venv)"
    return 0
  fi

  say "Creating Python environment (.venv)…"
  local py
  py="$(find_python)" || die "Python 3.${PYTHON_MIN_MINOR}+ not found. Install Python from https://www.python.org/downloads/ or: brew install python@3.12"

  if have uv; then
    run_logged uv venv --python "$py" --seed "$ROOT/.venv" || die "uv venv failed"
  else
    run_logged "$py" -m venv "$ROOT/.venv" || die "python -m venv failed"
  fi
  say "✓ Created .venv"
}

venv_python() {
  echo "$ROOT/.venv/bin/python"
}

pip_install() {
  if have uv; then
    run_logged uv pip install --python "$(venv_python)" "$@"
  else
    run_logged "$(venv_python)" -m pip install "$@"
  fi
}

deps_healthy() {
  local py
  py="$(venv_python)"
  "$py" - <<'PY' >/dev/null 2>&1
import importlib

for mod in ("fastapi", "uvicorn", "av", "PIL", "huggingface_hub", "ltx_pipelines_mlx", "ltx_core_mlx"):
    importlib.import_module(mod)
PY
}

install_deps() {
  local tag force="${1:-0}"
  tag="$(mlx_tag)"

  if [[ "$force" != "1" ]] && deps_healthy; then
    date >"$MARKER" 2>/dev/null || true
    say "✓ Dependencies already installed"
    return 0
  fi

  say "Installing Python packages (first run can take several minutes)…"
  pip_install -U pip setuptools wheel || die "Could not upgrade pip"
  pip_install -r "$ROOT/requirements.txt" || die "requirements.txt install failed"
  pip_install \
    "ltx-core-mlx @ git+https://github.com/dgrauet/ltx-2-mlx.git@${tag}#subdirectory=packages/ltx-core-mlx" \
    "ltx-pipelines-mlx @ git+https://github.com/dgrauet/ltx-2-mlx.git@${tag}#subdirectory=packages/ltx-pipelines-mlx" \
    || die "ltx-2-mlx install failed (need network + git)"

  if deps_healthy; then
    date >"$MARKER"
    say "✓ Dependencies installed"
  else
    die "Packages installed but imports still fail — see log above and $LOG_FILE"
  fi
}

ensure_web_ui() {
  if [[ -f "$ROOT/web/dist/index.html" ]]; then
    say "✓ Web UI build present"
    return 0
  fi

  if ! have npm; then
    die "Web UI is not built and Node.js/npm is missing.
Install Node from https://nodejs.org/ (LTS), then double-click this script again.
Or run once:  cd web && npm install && npm run build"
  fi

  say "Building Web UI (npm)…"
  (
    cd "$ROOT/web" || exit 1
    if [[ ! -d node_modules ]]; then
      run_logged npm install || exit 1
    fi
    run_logged npm run build || exit 1
  ) || die "Web UI build failed"
  say "✓ Web UI built"
}

wait_for_ui() {
  local i
  for i in $(seq 1 90); do
    if curl -fsS -o /dev/null --max-time 1 "$UI_URL" 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  return 1
}

open_browser_soon() {
  (
    if wait_for_ui; then
      say "Opening browser → $UI_URL"
      open "$UI_URL" 2>/dev/null || true
    else
      say "Server is starting slowly (model download on first run is normal)."
      say "Watch this window for download / load progress."
      say "When ready, open: $UI_URL"
      open "$UI_URL" 2>/dev/null || true
    fi
  ) &
}

show_crash_help() {
  local ec="$1"
  say ""
  say "══════════════════════════════════════════════════════"
  say "⚠ Server crashed (exit $ec)."
  say "  • Scroll up in this window for the error text"
  say "  • Or open the saved log: $LOG_FILE"
  say "══════════════════════════════════════════════════════"
  say "── last 30 log lines ──"
  tail -n 30 "$LOG_FILE" 2>/dev/null || true
  say "── end ──"
}

start_server_bg() {
  local py="$1"
  # Background so this shell keeps traps (Ctrl+C confirm / window-close kill).
  # Logs: Terminal + file, unbuffered.
  "$py" -u "$ROOT/server.py" --host 127.0.0.1 --port "$PORT" --model "$MODEL" \
    > >(tee -a "$LOG_FILE") 2>&1 &
  SERVER_PID=$!
  echo "$SERVER_PID" >"$PID_FILE"
  say "Server pid $SERVER_PID"
}

wait_for_server() {
  local ec=0
  while [[ -n "${SERVER_PID:-}" ]]; do
    set +e
    wait "$SERVER_PID" 2>/dev/null
    ec=$?
    set -u

    # User confirmed stop via Ctrl+C dialog.
    if [[ "$USER_STOP" == "1" ]]; then
      SERVER_PID=""
      rm -f "$PID_FILE"
      return 130
    fi

    # wait() can return >128 when a trapped signal fired (e.g. Ctrl+C → Keep running).
    # If the server is still alive, resume waiting.
    if [[ $ec -gt 128 ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
      continue
    fi

    SERVER_PID=""
    rm -f "$PID_FILE"
    return "$ec"
  done
  return 0
}

run_server_supervised() {
  local py backoff=2 crashes=0 ec
  py="$(venv_python)"

  say ""
  say "Starting LTX-WS…"
  say "  Model:  $MODEL"
  say "  Live logs appear below (also saved to disk)."
  say "  Leave this window open while you use the UI."
  say "  Ctrl+C asks before stopping · closing the window kills the server."
  say ""

  open_browser_soon

  while true; do
    [[ "$USER_STOP" == "1" ]] && break

    kill_dangling_ltx_ws
    start_server_bg "$py"
    wait_for_server
    ec=$?

    [[ "$USER_STOP" == "1" ]] && break

    # Clean stop
    if [[ $ec -eq 0 || $ec -eq 130 || $ec -eq 143 ]]; then
      say "Server stopped (exit $ec)."
      break
    fi

    crashes=$((crashes + 1))
    show_crash_help "$ec"
    say "Auto-restart #$crashes in ${backoff}s…"

    if ! deps_healthy; then
      say "Repairing Python packages…"
      install_deps 1 || true
    fi
    if [[ ! -f "$ROOT/web/dist/index.html" ]]; then
      ensure_web_ui || true
    fi

    sleep "$backoff"
    if [[ $backoff -lt $MAX_BACKOFF ]]; then
      backoff=$((backoff * 2))
      if [[ $backoff -gt $MAX_BACKOFF ]]; then
        backoff=$MAX_BACKOFF
      fi
    fi
    say ""
    say "Restarting server (attempt after $crashes crash(es))…"
    say ""
  done
}

# ── main ──────────────────────────────────────────────────────────────
banner

if [[ "$(uname -s)" != "Darwin" ]]; then
  die "This launcher is for macOS. On other systems, run: python server.py"
fi

arch="$(uname -m)"
if [[ "$arch" != "arm64" ]]; then
  say "WARNING: Apple Silicon (arm64) is required for MLX. This Mac reports: $arch"
  say "Generation will likely fail. Continuing setup anyway…"
  say ""
fi

[[ -f "$ROOT/server.py" ]] || die "server.py not found. Put this script in the ltx-ws folder."
[[ -f "$ROOT/requirements.txt" ]] || die "requirements.txt not found."

ensure_venv
install_deps
ensure_web_ui
say "Checking for leftover LTX-WS processes…"
kill_dangling_ltx_ws
run_server_supervised
