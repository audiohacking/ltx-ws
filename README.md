# ltx-ws

**Local LTX-2.3 video over WebSocket on Apple Silicon (MLX).** This repository provides:

| Component | Role |
|-----------|------|
| [`server.py`](server.py) | WebSocket server: loads [ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx), runs **T2V/I2V/A2V/retake/extend**, streams **MP4** to clients. |
| [`videofentanyl.py`](videofentanyl.py) | CLI client: queues jobs, speaks the same JSON + binary protocol (`--mode ltx` + `--server`). |
| [`mcp_server.py`](mcp_server.py) | MCP server exposing standardized tools for LTX generation (`ltx_generate_video`, `ltx_server_healthcheck`). |
| [`web_ui.py`](web_ui.py) | Web UI API + static assets; embedded in `server.py` by default (`--web-ui`). |
| [`web/`](web/) | React frontend — **LTX-WS Videofentanyl** (player, library, multi-clip autocontinue/autoconcat, LoRA picker). |
| [`web_server.py`](web_server.py) | Optional standalone UI when attaching to a remote WebSocket server. |
| [`ltx_mlx_backend.py`](ltx_mlx_backend.py) | MLX pipeline adapter, Hugging Face weight resolution, frame/spatial alignment. |
| [`scripts/benchmark_local_generation.py`](scripts/benchmark_local_generation.py) | Spawns (or attaches to) `server.py`, runs one client job, prints timings + `BENCHMARK_JSON:…`. |

Everything below is **local-only**: your Mac, Metal / MLX, and optional Hugging Face downloads. No hosted inference is required.

---

## Features

- **Agent-ready docs** — [`AGENTS.md`](AGENTS.md) and [`CLAUDE.md`](CLAUDE.md) document MCP purpose, tools, and continuity (`autocontinue`) usage for AI coding agents.
- **MLX on Metal** — Inference via [`ltx_pipelines_mlx`](https://github.com/dgrauet/ltx-2-mlx) **v0.14.9** (`DistilledPipeline`, `TI2VidOneStagePipeline` for i2v/autocontinue, `A2VidPipelineTwoStage`, `RetakePipeline`, etc.). Legacy separate `ImageToVideoPipeline` exists only on older ltx-2-mlx installs.
- **Automatic weight download** — For a Hugging Face repo id (`org/model`), the server calls [`huggingface_hub.snapshot_download`](https://huggingface.co/docs/huggingface_hub/guides/download) on load (equivalent to `huggingface-cli download`). Resumes partial downloads.
- **Default weights** — [`dgrauet/ltx-2.3-mlx`](https://huggingface.co/dgrauet/ltx-2.3-mlx) (full MLX bf16; very large). Use [`ltx-2.3-mlx-q8`](https://huggingface.co/dgrauet/ltx-2.3-mlx-q8) or [`-q4`](https://huggingface.co/dgrauet/ltx-2.3-mlx-q4) for less RAM/disk.
- **Weight paths** — `./models/<org>__<name>/` by default, or `--model-dir`, or base directory `$VIDEOFENTANYL_MODELS`.
- **Per-job overrides** — Client `simple_generate` may send `seed`, `num_frames`, `height`, `width`, `num_steps` (server snaps frames to **8k+1** and resolution to **multiples of 32**).
- **Image/audio/video inputs** — Session / generate payloads support image keys plus `audio_input` and `source_video`; client supports `--image`, `--audio`, `--video` (path or `http(s)` URL).
- **Cross-machine safe media upload** — Client serializes `--image`, `--audio`, and `--video` as payload data, so server and client can run on different hosts without shared filesystem paths.
- **Operation routing** — `--generation-mode generate|a2v|retake|extend` maps to matching MLX pipelines, including `--retake-start`, `--retake-end`, `--extend-frames`, `--extend-direction`.
- **Long runs without stalling** — Server emits `generation_keepalive` JSON during inference (includes **denoising step progress** when available); client may send `generation_status` and receive `generation_status_ack`.
- **Disconnect safety** — Finished MP4s are copied to `--spill-dir` if the client drops while streaming (`fvserver_completed` by default).
- **Single-flight generation** — One active MLX job at a time; extra clients wait in a fair queue (`queue_status` / `gpu_assigned`).
- **Client batching** — Multiple `--prompt`s, `--count`, `--delay`, `--retries`, `--dry-run`, `--verbose`.
- **Autocontinue / autoconcat** — Chain clips using the last frame of each as the next start image; optional `ffmpeg` stream-copy merge into one file (`--autoconcat`).
- **Embedded Web UI** — **LTX-WS Videofentanyl** in the browser (`http://<host>:8765/`): prompt bar, clip library, denoising progress (step + ETA), multi-clip **autocontinue** + **autoconcat**, and **LoRA** dropdown (default OmniNFT RL; auto-downloads to `./loras/`).
- **LoRA** — CLI `--enable-lora` for global defaults, or per-request via Web UI / `lora_specs` in API/MCP. Default artifact: [OmniNFT RL LoRA](https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors) (`DEFAULT_LORA_URL` / `LTX_WS_DEFAULT_LORA`).

---

## Requirements

| | |
|--|--|
| **Hardware** | Apple Silicon (M1 / M2 / M3 / M4 …). |
| **OS** | macOS (Metal). |
| **Python** | **3.11+** recommended for `server.py` and ltx-2-mlx; 3.10+ may work for the client only. |
| **ffmpeg** | Optional; required on the **client** machine if you use `--autoconcat`. |
| **Disk / RAM** | Depends on model (bf16 ≫ q8 ≫ q4). Plan tens of GB disk for full bf16 weights; see [ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx) model table. |

Python packages: see [`requirements.txt`](requirements.txt) (`websockets`, `av`, `Pillow`, `huggingface_hub`, `fastapi`, `starlette`, `uvicorn`, `python-multipart`, `mcp`). **MLX** packages are installed separately from the ltx-2-mlx monorepo (comments in `requirements.txt`).

Optional: [uv](https://docs.astral.sh/uv/) for fast venv + installs; [Node.js](https://nodejs.org/) 18+ to build the Web UI.

---

## Quick start

From the repository root. If you **already have weights** under `./models/`, the server will use them — nothing is wiped on `git pull` (see [Model weights](#model-weights)).

```bash
cd ltx-ws

# 1) Python environment — pick uv (recommended) or classic venv (below)

uv venv --python 3.12 --seed
source .venv/bin/activate          # Windows: .venv\Scripts\activate

uv pip install -r requirements.txt
uv pip install \
  "ltx-core-mlx @ git+https://github.com/dgrauet/ltx-2-mlx.git@v0.14.9#subdirectory=packages/ltx-core-mlx" \
  "ltx-pipelines-mlx @ git+https://github.com/dgrauet/ltx-2-mlx.git@v0.14.9#subdirectory=packages/ltx-pipelines-mlx"

# 2) Web UI (first time, or after UI changes)
cd web && npm install && npm run build && cd ..

# 3) Point at weights you already have (list ./models/ and match the folder name)
ls models/
python server.py --model ltx-2.3-mlx-q8

# 4) Browser → http://127.0.0.1:8765/   WebSocket → ws://127.0.0.1:8765/ws
```

CLI client (separate terminal, same venv):

```bash
source .venv/bin/activate
python videofentanyl.py --server ws://127.0.0.1:8765/ws --prompt "a fox running through snow"
```

---

## Install

### Option A — `uv` (recommended)

[`uv`](https://docs.astral.sh/uv/) creates the virtualenv and installs packages quickly.

```bash
git clone https://github.com/lmangani/ltx-ws.git
cd ltx-ws

uv venv --python 3.12 --seed
source .venv/bin/activate   # fish: source .venv/bin/activate.fish

uv pip install -r requirements.txt
uv pip install \
  "ltx-core-mlx @ git+https://github.com/dgrauet/ltx-2-mlx.git@v0.14.9#subdirectory=packages/ltx-core-mlx" \
  "ltx-pipelines-mlx @ git+https://github.com/dgrauet/ltx-2-mlx.git@v0.14.9#subdirectory=packages/ltx-pipelines-mlx"
```

Later sessions — reactivate the same env (your `models/` folder is untouched):

```bash
cd ltx-ws
source .venv/bin/activate
python server.py --model ltx-2.3-mlx-q8
```

### Option B — `venv` + `pip`

```bash
git clone https://github.com/lmangani/ltx-ws.git
cd ltx-ws

python3.12 -m venv .venv
source .venv/bin/activate   # fish: source .venv/bin/activate.fish

pip install -U pip
pip install -r requirements.txt
pip install \
  "ltx-core-mlx @ git+https://github.com/dgrauet/ltx-2-mlx.git@v0.14.9#subdirectory=packages/ltx-core-mlx" \
  "ltx-pipelines-mlx @ git+https://github.com/dgrauet/ltx-2-mlx.git@v0.14.9#subdirectory=packages/ltx-pipelines-mlx"
```

### Web UI assets

Build once (or after editing `web/`):

```bash
cd web && npm install && npm run build && cd ..
```

### macOS app (PyInstaller)

Build a double-clickable **LTX-WS Videofentanyl** `.app` (no terminal; status in the Web UI header). See [docs/PACKAGING.md](docs/PACKAGING.md).

**Download:** published GitHub Releases include `LTX-WS-Videofentanyl-<tag>-macos-arm64.zip` (built by [.github/workflows/release.yml](.github/workflows/release.yml)).

**Build locally:**

```bash
./scripts/ci_install_build_deps.sh
./scripts/build_mac_app.sh
```

Output: `dist/LTX-WS Videofentanyl.app`. Models and outputs live under `~/Library/Application Support/LTX-WS/`.

### Hugging Face auth

For gated or private Hub repos: set [`HF_TOKEN`](https://huggingface.co/docs/huggingface_hub/package_reference/environment_variables) or run `huggingface-cli login`.

---

## Model weights

**MLX only.** This server uses **ltx-2-mlx** (`ltx_pipelines_mlx`) on Apple Silicon. You must use **MLX-converted** checkpoints — **not** standard upstream LTX 2.3 weights.

| Use | Do **not** use |
|-----|----------------|
| `dgrauet/ltx-2.3-mlx`, `dgrauet/ltx-2.3-mlx-q8`, `dgrauet/ltx-2.3-mlx-q4` | `Lightricks/LTX-2.3` (PyTorch / diffusers layout) |
| Local dirs from `snapshot_download` of the **dgrauet** MLX repos above | `Lightricks/LTX-2` or other non-MLX Hub repos |
| `--model auto` (picks a **dgrauet** MLX variant by RAM) | ComfyUI / CUDA checkpoints without MLX conversion |

See [AGENTS.md](AGENTS.md) for the full agent rule: **only ever MLX weights from the ltx-2-mlx ecosystem.**

### Keeping your `./models/` folder

- Weights live in **`./models/`** at the repo root (or paths you pass to `--model` / `--model-dir`).
- The directory is **gitignored** so large checkpoints are **not** committed or removed by `git clone` / `git pull` — they stay on your machine between updates.
- The server **does not delete** `models/`; it only reads weights or downloads missing Hub snapshots into that tree.
- After pulling new code, reactivate `.venv` and run `server.py` with the same `--model` shorthand you used before.

**If you already have models** (typical layout):

```bash
ls models/
# e.g. ltx-2.3-mlx-q8  or  dgrauet__ltx-2.3-mlx-q8

python server.py --model ltx-2.3-mlx-q8
# or explicit path:
python server.py --model ./models/ltx-2.3-mlx-q8
```

**First-time download** — pass a Hugging Face repo id; snapshots go under `./models/<org>__<name>/` (unless `--model-dir` or `$VIDEOFENTANYL_MODELS` applies).

**Local directory** — pass any existing MLX weights directory to `--model` instead of `org/name`.

**Single-folder names (e.g. `./models/ltx-2.3-mlx/`)**

Hugging Face ids look like `author/model` (with a **slash**). A bare name like `ltx-2.3-mlx` is **not** a Hub id; if it is passed through unchanged, `ltx_pipelines_mlx` may try `https://huggingface.co/ltx-2.3-mlx` and fail with **404**.

The server resolves local weights in this order before any Hub download:

1. `--model` is an existing directory path (relative or absolute). For **relative** paths it checks **the current working directory first**, then the **repository root** (the folder that contains `ltx_mlx_backend.py`), so you can start the server from another directory and still use `./models/...` next to the checkout.
2. **`--model-dir`** uses the same rule (cwd, then repo root) when the path is relative.
3. **`<repo>/models/<name>/`** and **`./models/<name>/` from cwd** for a shorthand leaf name (no `/` in `<name>`) — then `--model <name>` alone is enough if one of those folders exists.

**RAM-based default (`--model` omitted or `auto`)**

If you do not pass `--model`, the server defaults to **`auto`**: it reads total physical RAM (on macOS, `sysctl hw.memsize`; Apple Silicon uses **unified memory**, so this is the same pool MLX uses—there is no separate VRAM to probe) and picks a pre-converted MLX repo:

| Variant | Hugging Face repo | Approx. weights | Auto when RAM is |
|--------|-------------------|-----------------|------------------|
| bf16 | `dgrauet/ltx-2.3-mlx` | ~42 GB | **≥ 64 GiB** |
| int8 | `dgrauet/ltx-2.3-mlx-q8` | ~21 GiB | **≥ 32 GiB** and under 64 GiB |
| int4 | `dgrauet/ltx-2.3-mlx-q4` | ~12 GiB | **under 32 GiB** (still chosen if RAM is below 16 GiB, with a startup warning) |

Pass an explicit **`--model <repo or path>`** to skip auto-selection. You can also set **`LTX_WS_MODEL`** to the default when the flag is omitted (e.g. `LTX_WS_MODEL=dgrauet/ltx-2.3-mlx-q8` or `LTX_WS_MODEL=auto`).

**Practical defaults**

```bash
# Same as omitting --model: resolve from installed RAM
python server.py --model auto

# Smaller quantised model (recommended for many machines)
python server.py --model dgrauet/ltx-2.3-mlx-q8

# Explicit download directory
python server.py --model dgrauet/ltx-2.3-mlx-q4 --model-dir "$HOME/mlx-weights/ltx-q4"

# Custom snapshot folder under ./models/ (any of these work if the directory exists)
python server.py --model ./models/ltx-2.3-mlx
python server.py --model ltx-2.3-mlx
python server.py --model ltx-2.3-mlx --model-dir ./models/ltx-2.3-mlx
```

---

## Run the server

With `.venv` activated:

```bash
source .venv/bin/activate
python server.py --model ltx-2.3-mlx-q8
```

By default the **Web UI is enabled** on the same port:

| Service | URL |
|---------|-----|
| Web UI | `http://127.0.0.1:8765/` |
| WebSocket | `ws://127.0.0.1:8765/ws` |

WebSocket-only (no browser UI):

```bash
python server.py --no-web-ui --model ltx-2.3-mlx-q8
```

Model path and pipelines are resolved at startup; the first use of each pipeline (`t2v` / `i2v` / `a2v` / `retake` / `extend`) is lazy-loaded.

Useful variants:

```bash
python server.py --port 9000
python server.py --model dgrauet/ltx-2.3-mlx-q8 --infer-steps 8 --num-frames 65
python server.py --height 512 --width 768 --mlx-low-memory
python server.py --upscale --height 768 --width 1344
python server.py --enable-lora \
  --lora https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors 1.0
# multiple LoRAs (repeat --lora)
python server.py --enable-lora \
  --lora https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors 1.0 \
  --lora /path/to/another_lora.safetensors 0.6
# enable default LoRA via env
LTX_WS_ENABLE_LORA=1 python server.py
# override default LoRA via env (still requires enable)
LTX_WS_DEFAULT_LORA="https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors" \
LTX_WS_DEFAULT_LORA_SCALE="1.0" LTX_WS_ENABLE_LORA=1 python server.py
# multi-default via env (comma-separated path:scale)
LTX_WS_DEFAULT_LORAS="https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors:1.0,/path/to/another_lora.safetensors:0.6" \
LTX_WS_ENABLE_LORA=1 python server.py
```

With `--upscale`, `ltx-ws` now runs a true two-stage generate path: stage 1 at half-resolution, then a spatial upscaler second stage where a tiled sampler is requested when supported by your installed `ltx-2-mlx` version.

## Run the MCP server

After `server.py` is running (with `.venv` activated):

```bash
source .venv/bin/activate
python mcp_server.py --server-url ws://127.0.0.1:8765/ws
```

This MCP server exposes:
- `ltx_server_healthcheck` — verify that your local `ltx-ws` endpoint is reachable.
- `ltx_generate_video` — run one generation job (supports `generate`, `a2v`, `retake`, `extend`, `ic_lora`) and return output path + timing metadata.
- `ltx_generate_sequence` — run a prompt list as chained clips with `autocontinue` support (last frame of clip N is fed as initial image to clip N+1), with optional `autoconcat`.

## Web UI (LTX-WS Videofentanyl)

Browser client served from `server.py` by default (`--web-ui`, same port as WebSocket). Title in the header: **LTX-WS Videofentanyl**.

| Feature | Notes |
|---------|--------|
| Player + library | Completed clips stay in the library when you start a new generation; select a clip to restore its settings. |
| Multi-clip | **Clips (× duration)** sets count; autocontinue + autoconcat run automatically for ×N > 1 (same as CLI `--count N --autocontinue --autoconcat`). |
| Progress | Denoising step counter and ETA during generation (SSE from embedded worker). |
| LoRA dropdown | Default **OmniNFT RL LoRA** (`DEFAULT_LORA_URL` / `LTX_WS_DEFAULT_LORA`); downloads on first use via `/api/loras/ensure`. **None** disables LoRA for that job. No `--enable-lora` required for UI per-request LoRA. |
| Modes | generate, i2v, a2v, retake, extend, ic_lora (with uploads as needed). |

Embedded in `server.py` by default — no separate process. See [Quick start](#quick-start) for build + run.

Open from another machine on the LAN using the host IP (server binds `0.0.0.0` by default): `http://<your-mac-ip>:8765/`.

**Development** (hot-reload frontend while `server.py` runs):

```bash
source .venv/bin/activate
python server.py --model ltx-2.3-mlx-q8
cd web && npm run dev   # :5299, proxies /api and /ws → :8765
```

**Standalone UI** (attach to a remote WebSocket server):

```bash
source .venv/bin/activate
python web_server.py --server-url ws://127.0.0.1:8765/ws
```

Generated clips persist under `./web_outputs/` (override with `--web-output-dir` on `server.py`).

### OmniNFT LoRA (default)

Canonical default URL (also `DEFAULT_LORA_URL` in `server.py`):

- `https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors`

**Web UI:** LoRA dropdown defaults to this preset; first use triggers download/cache under `./loras/`.

**CLI / global server default** (applied to every request when enabled):

```bash
python server.py --enable-lora \
  --lora https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors 1.0
```

**Per-request only** (no `--enable-lora`): pass `lora_specs` in `/api/generate`, MCP, or use the Web UI dropdown.

Override default via env (still used for Web UI catalog when set):

```bash
export LTX_WS_DEFAULT_LORA="https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors"
export LTX_WS_DEFAULT_LORA_SCALE="1.0"
```

Other methods (manual download):

**huggingface-cli:**

```bash
mkdir -p ./loras/Kijai__LTX2.3_comfy
huggingface-cli download Kijai/LTX2.3_comfy \
  --include "loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors" \
  --local-dir ./loras/Kijai__LTX2.3_comfy

python server.py --enable-lora \
  --lora ./loras/Kijai__LTX2.3_comfy/loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors 1.0
```

**curl:**

```bash
mkdir -p ./loras
curl -L \
  "https://huggingface.co/Kijai/LTX2.3_comfy/resolve/main/loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors" \
  -o ./loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors

python server.py --enable-lora \
  --lora ./loras/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors 1.0
```

Downloaded LoRAs are cached under `./loras/` by default. Override with `VIDEOFENTANYL_LORA_DIR`.

---

## Run the client (local MLX)

`videofentanyl.py` must use **`--mode ltx`** (default) and **`--server`** pointing at your `server.py` WebSocket URL.

```bash
python videofentanyl.py --server ws://127.0.0.1:8765/ws --prompt "a fox running through snow"

python videofentanyl.py --server ws://127.0.0.1:8765/ws --prompt "sunset" --count 3

python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --prompt "animate gently" --image ./photo.jpg

python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --generation-mode a2v --audio ./music.wav --prompt "a musician performing"

python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --generation-mode retake --video ./source.mp4 --retake-start 1 --retake-end 3 \
  --prompt "A different scene"

python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --generation-mode extend --video ./source.mp4 --extend-frames 2 --extend-direction after \
  --prompt "Continue the motion"

python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --generation-mode a2v --audio ./song.wav --count 5 \
  --num-frames 121 --audiocontinue --prompt "music video"

python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --prompt "a river in a canyon" --count 4 --autocontinue

python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --prompt "timelapse" --count 3 --autocontinue --autoconcat
```

### Duration and aspect-ratio examples

`--num-frames` controls clip duration (`seconds ≈ frames / 24`).  
Use `8k+1` frame counts (for example: `49`, `97`, `121`, `193`).

```bash
# ~2.0s clip (49 / 24)
python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --prompt "cinematic close-up of a singer" --num-frames 49

# ~4.0s clip (97 / 24)
python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --prompt "a fox running through snow" --num-frames 97

# ~5.0s clip (121 / 24)
python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --prompt "street dance performance at night" --num-frames 121
```

Social/vertical outputs are controlled with `--height` and `--width` (server snaps to multiples of 32):

```bash
# 9:16 vertical (stories/reels style), text-to-video
python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --prompt "fashion model walking through neon alley" \
  --height 1024 --width 576 --num-frames 97

# 9:16 vertical with starting image (image-to-video)
python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --prompt "slow cinematic movement, subtle wind in hair" \
  --image ./vertical_start.jpg \
  --height 1024 --width 576 --num-frames 121

# 4:5 vertical post format with starting image
python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --prompt "product reveal with soft studio lighting" \
  --image ./post_4x5.jpg \
  --height 960 --width 768 --num-frames 97

# 1:1 square format
python videofentanyl.py --server ws://127.0.0.1:8765/ws \
  --prompt "loopable abstract animation" \
  --height 768 --width 768 --num-frames 97
```

Tip: when using a starting image, match `--height/--width` to the image orientation for best framing (portrait source image + portrait output).

**Dry-run** (print queue, no network):

```bash
python videofentanyl.py --server ws://127.0.0.1:8765/ws --prompt "test" --count 5 --dry-run
```

---

## Benchmark

From the repo root (uses `.venv/bin/python3` when present):

```bash
./scripts/benchmark_local_generation.py
./scripts/benchmark_local_generation.py --port 9000 --no-server --server-url ws://studio.local:9000/ws
```

The last line of output is **`BENCHMARK_JSON:{...}`** for scripts. Outputs go under `benchmark_runs/` by default.

---

## Repository layout

```
server.py                 # WebSocket + embedded Web UI (default)
videofentanyl.py          # CLI client
mcp_server.py             # MCP tool server
web_ui.py                 # Web UI API / orchestration
web_server.py             # Standalone Web UI entry point
web/                      # React frontend (build → web/dist/)
ltx_mlx_backend.py        # MLX generator + HF snapshot paths
requirements.txt
scripts/benchmark_local_generation.py
models/                   # local MLX weights (gitignored — kept on disk)
web_outputs/              # Web UI generated clips (gitignored)
third_party/LTX-2/        # optional submodule
```

---

## WebSocket protocol (local server)

JSON messages use a `type` field. After `simple_generate`, the server streams **raw MP4 bytes** in chunks (`--chunk-size` on the server). Typical sequence:

```
client →  session_init_v2          (session + optional initial image for i2v / autocontinue)
client →  simple_generate         (prompt + mode; optional seed/frames/size/steps + image/audio/video keys)
server ←  queue_status             (while waiting for the single MLX slot)
server ←  gpu_assigned             (generation_id, gpu_id reports mlx:0)
server ←  ltx2_stream_start       (single-segment stream)
server ←  ltx2_segment_start
server ↔  generation_keepalive     (periodic JSON during inference)
client →  generation_status       (optional)
server ←  generation_status_ack   (phase, elapsed_s, generation_id)
server ←  [binary MP4 chunks]
server ←  ltx2_segment_complete
server ←  ltx2_stream_complete
server ←  latency                 (timing metadata)
```

The client implements this flow for `--mode ltx` when `--server` is set.

---

## `server.py` CLI reference

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `0.0.0.0` | Bind address. |
| `--port` | `8765` | TCP port; path **`/ws`**. |
| `--model` | `auto` (or `$LTX_WS_MODEL`) | HF repo id, local weights directory, or **`auto`** (RAM → bf16 / q8 / q4; see [Model weights](#model-weights)). |
| `--model-dir` | *(see Models)* | Store HF snapshot here; overrides default `./models/<org>__<name>/`. |
| `--enable-lora` | off | Enable global LoRA handling on the server. |
| `--lora <path_or_repo_or_url> <scale>` | off unless enabled | Global LoRA(s) applied to all requests; **repeat flag** to stack multiple LoRAs. |
| `--num-frames` | `97` | Target length; adjusted to **8k+1** (e.g. 9, 25, 49, 97). |
| `--height` | `480` | Snapped to multiple of **32**. |
| `--width` | `704` | Snapped to multiple of **32**. |
| `--fps` | `24` | Nominal rate (mux behaviour follows pipeline). |
| `--infer-steps` | `8` | One-stage distilled step count (minimum 1). |
| `--upscale` | off | `generate` mode only: stage 1 at ½ resolution, then spatial upscaler second stage to final size; requests tiled sampler for stage 2 when backend supports it. |
| `--mlx-low-memory` | off | `low_memory=True` in ltx-2-mlx (slower, less RAM). |
| `--chunk-size` | `65536` | Max bytes per WebSocket binary frame. |
| `--spill-dir` | `fvserver_completed` | Salvage directory on client disconnect. |
| `--verbose` | off | Extra per-connection logs. |
| `--web-ui` | on | Serve browser UI + `/api` on the same port (use `--no-web-ui` to disable). |
| `--web-output-dir` | `./web_outputs` | Directory for Web UI saved clips. |

Default global LoRA is **disabled unless enabled** with `--enable-lora` (or env below).  
When enabled, LoRA defaults can be configured in `server.py` constants and overridden via env:
- `LTX_WS_ENABLE_LORA`
- `LTX_WS_DEFAULT_LORA`
- `LTX_WS_DEFAULT_LORA_SCALE`
- `LTX_WS_DEFAULT_LORAS` (comma-separated `path:scale,path:scale`) for multiple defaults

LoRA artifacts are resolved from local path, URL, or Hugging Face repo id. Downloaded LoRAs are cached under `./loras/` by default; override with:
- `VIDEOFENTANYL_LORA_DIR`
When LoRA is enabled, server startup now pre-resolves/downloads global LoRAs (fail-fast), matching the main model preflight behavior.

---

## `videofentanyl.py` CLI reference (local `--server`)

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | `ltx` | Use `ltx` for this stack (**requires `--server`**). |
| `--server` | — | WebSocket URL, e.g. `ws://127.0.0.1:8765/ws`. |
| `--prompt` / `-p` | *(built-in demo)* | Repeat for multiple prompts. |
| `--count` / `-n` | `1` | Videos per prompt. |
| `--seed`, `--num-frames`, `--height`, `--width`, `--num-steps` | — | Per-job generation overrides for local server. |
| `--generation-mode` | `generate` | Local route: `generate`, `a2v`, `retake`, `extend`. |
| `--image` / `-i` | — | Image-to-video: path or `http(s)` URL. |
| `--audio` | — | Audio-to-video input for `--generation-mode a2v`. |
| `--video` | — | Source video for `--generation-mode retake|extend`. |
| `--retake-start`, `--retake-end` | — | Retake frame range for `--generation-mode retake`. |
| `--extend-frames`, `--extend-direction` | — | Extend parameters for `--generation-mode extend`. |
| `--enhance` / `-e` | off | Sets `enhancement_enabled` in the client handshake; **this MLX server does not run GPT rewrite** — generation uses the prompt you send. |
| `--preset-id`, `--preset-label` | — | Override `session_init_v2` preset fields. |
| `--auto-extension`, `--loop` | off | Forwarded session flags. |
| `--output-dir` / `-o` | `.` | Save directory. |
| `--prefix` | `ltx` | Filename prefix. |
| `--ext` | `mp4` | Extension. |
| `--delay` / `-d` | `1.0` | Seconds between jobs. |
| `--retries` / `-r` | `1` | Retries per job. |
| `--idle-timeout` | unlimited with `--server` | Seconds of silence before a WebSocket ping probe. |
| `--verbose` / `-v` | off | Full protocol trace. |
| `--dry-run` | off | Print plan, exit. |
| `--autocontinue` | off | Last frame → next job’s start image. |
| `--autoconcat` | off | After queue: `ffmpeg -c copy` merge (**requires** `--autocontinue` + ffmpeg). |
| `--audiocontinue` | off | Music-video helper for `a2v`: implies `--autocontinue --autoconcat --autocompact`, splits `--audio` per clip and assigns one segment per job. |

Saved files look like: **`{prefix}_{NNN}_{slug}_{timestamp}.mp4`**. After a successful **`--autoconcat`**, fragments are removed and **`{prefix}_merged_{timestamp}.mp4`** is written.

---

## Troubleshooting

| Issue | What to do |
|-------|------------|
| **Player won’t open MP4** | Fragments are progressive / fMP4-style; remux: `ffmpeg -i in.mp4 -c copy out.mp4`. |
| **`Missing ltx_pipelines_mlx`** | Install the two `uv pip install …ltx-2-mlx.git#subdirectory=…` lines from `requirements.txt`. |
| **`huggingface_hub` errors** | Install deps; check `HF_TOKEN` for gated models; ensure enough free disk. |
| **OOM / slow** | Use `--model dgrauet/ltx-2.3-mlx-q8` or `-q4`, lower `--num-frames` / resolution, or `--mlx-low-memory`. |
| **Port already in use** | `--port` on server and matching URL on client. |
| **`autoconcat` failed** | Install `ffmpeg` on the client host; fragments are kept if merge fails. |

---

## References

- Repo: [github.com/lmangani/ltx-ws](https://github.com/lmangani/ltx-ws)  
- Inference stack: [github.com/dgrauet/ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx)  
- Default weights: [huggingface.co/dgrauet/ltx-2.3-mlx](https://huggingface.co/dgrauet/ltx-2.3-mlx)  
- Hub downloads: [huggingface.co/docs/huggingface_hub/guides/download](https://huggingface.co/docs/huggingface_hub/guides/download)
