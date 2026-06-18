# Training UI plan (`/train`)

Research-backed plan for a **ltx-ws** training lab on top of [ltx-2-mlx](https://github.com/dgrauet/ltx-2-mlx) **v0.14.12** (`ltx-trainer-mlx`).

## What “training” means upstream

ltx-2-mlx is a three-package monorepo:

| Package | Role |
|---------|------|
| `ltx-core-mlx` | Model weights, VAE, DiT, Gemma connectors |
| `ltx-pipelines-mlx` | Inference CLI (`generate`, `retake`, …) |
| `ltx-trainer-mlx` | **LoRA / full fine-tune** via flow matching |

Training is **not** online learning during inference. It is an offline pipeline:

```
raw videos  →  [slice]  →  clips + captions  →  [preprocess]  →  latents + conditions  →  [train]  →  LoRA .safetensors
```

### CLI entry points (from `ltx_pipelines_mlx/cli.py`)

All require optional package install:

```bash
uv pip install \
  "ltx-trainer-mlx @ git+https://github.com/dgrauet/ltx-2-mlx.git@v0.14.12#subdirectory=packages/ltx-trainer"
```

| Command | Python API | Purpose |
|---------|------------|---------|
| `ltx-2-mlx slice` | `ltx_trainer_mlx.slice_clips.slice_videos` | Cut long sources into fixed-length, 32-aligned clips (ffmpeg; audio retained) |
| `ltx-2-mlx preprocess` | `ltx_trainer_mlx.preprocess.preprocess_dataset` | Encode clips → `.precomputed/latents/`, `conditions/`, optional `audio_latents/` |
| `ltx-2-mlx train` | `LtxvTrainer(config).train()` | Flow-matching LoRA (or full) training from preprocessed data |

### Training strategies (validated in `tests/test_trainer_core.py`)

| Strategy | Config `training_strategy.name` | Notes |
|----------|----------------------------------|-------|
| Text-to-video LoRA | `text_to_video` | Default; `generate_audio: false` for video-only style |
| Joint AV LoRA | `text_to_video` + `generate_audio: true` | Needs `preprocess --with-audio`; v0.14.12 audio path |
| Video-to-video (IC-LoRA) | `video_to_video` | Requires reference latents in preprocessed data; LoRA only |

Example configs ship in upstream `packages/ltx-trainer/configs/`:

- `lora_t2v.yaml` — basic T2V style LoRA
- `lora_v2v.yaml` — IC-LoRA / reference-video conditioning
- `lora_av_whisper.yaml` — joint audio+video (whisper/ASMR); uses `transformer-dev.safetensors`, gradient checkpointing

### Preprocessed data layout

```
<data_root>/
  .precomputed/
    latents/latent_0000.safetensors      # video VAE latents + dims/fps
    conditions/condition_0000.safetensors # Gemma prompt embeds
    audio_latents/latent_0000.safetensors # optional; paired filenames
```

Captions: sibling `.txt` per clip (or `--captions` dir with matching stems).

### Training runtime characteristics

- **Single-device MLX** on Apple Silicon (unified memory); no DDP.
- **Heavy RAM**: dev transformer + Gemma + activations. `enable_gradient_checkpointing` / CLI `--low-ram` needed on ≤64 GB for dev-base LoRAs.
- **Long-running**: thousands of steps; checkpoints + validation renders on interval.
- **Outputs**: `output_dir/` with checkpoints (`.safetensors`), validation MP4s, saved YAML config.
- **Progress hook**: `LtxvTrainer.train(step_callback=fn)` — `(global_step, total_steps, validation_paths)`.
- **Conflicts with inference**: training and generation both want GPU/RAM; must not run concurrently with `server.py` generation lock.

### Hardware guidance (from upstream configs + changelog)

| Workflow | Typical RAM | Resolution / frames |
|----------|-------------|-------------------|
| T2V LoRA (distilled base) | 32–48 GB | 704×480 × 25 frames validation |
| AV style LoRA (dev base + checkpointing) | 64 GB | 192×192 × 97 frames |
| Preprocess only | ~16 GB peak | Encoder + Gemma partial download (v0.14.12) |

Frame counts must stay **8k+1**; spatial dims **÷32**; training fps should stay near **24** (LTX training distribution).

---

## Gap in ltx-ws today

- Inference stack only: `ltx_mlx_backend.py`, `/api/generate`, main React UI.
- LoRA **inference** presets exist; no slice/preprocess/train orchestration.
- `ltx-trainer-mlx` not in `requirements.txt` (optional extra).
- Single generation worker; no training job queue.

---

## `/train` page — product goals

**Experimentation lab**, not a full MLOps platform:

1. Prepare a small dataset (upload or point at folder).
2. Run preprocess with sensible defaults.
3. Configure and launch a LoRA run (T2V first; AV/V2V later).
4. Watch step progress + validation previews.
5. Register finished LoRA into existing Web UI preset list for inference smoke tests.

---

## Proposed architecture

### Frontend (`web/`)

Add client routing (e.g. `react-router-dom`):

| Route | Page |
|-------|------|
| `/` | Existing generator (`App.tsx` → `GeneratePage`) |
| `/train` | New `TrainPage.tsx` |

`TrainPage` sections (wizard or tabs):

1. **Dataset** — upload videos + caption `.txt`; or path to local folder; optional slice settings (interval, res, caption template).
2. **Preprocess** — model id, H×W, max frames, `with_audio`, frame rate; start + progress.
3. **Train** — preset picker (T2V / AV / V2V), hyperparams (rank, steps, LR, checkpoint/val intervals), validation prompts; advanced YAML toggle.
4. **Runs** — list jobs, live step/loss, validation video thumbnails, cancel, download LoRA, **“Add to LoRA library”**.

Nav link in header: `Generate` | `Train`.

Vite: `historyApiFallback` already handled by FastAPI `html=True` static mount.

### Backend (`web_ui.py` + new `ltx_train_backend.py`)

Optional dependency gate: if `ltx_trainer_mlx` missing, `/api/train/health` returns `ok: false` with install hint.

```
web_outputs/
  train/
    <job_id>/
      uploads/          # raw uploads
      clips/            # post-slice
      preprocessed/     # .precomputed
      outputs/          # trainer output_dir
      config.yaml       # resolved trainer config
      status.json       # phase, step, logs
```

**API sketch:**

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/train/health` | Trainer package installed? ffmpeg? active MLX model path? |
| GET | `/api/train/presets` | Built-in config templates (T2V / AV / V2V) |
| POST | `/api/train/datasets` | Create dataset job dir; accept multipart uploads |
| POST | `/api/train/slice` | `{ dataset_id, interval, res, ... }` |
| POST | `/api/train/preprocess` | `{ dataset_id, model, height, width, with_audio, ... }` |
| POST | `/api/train/runs` | Build `LtxTrainerConfig`, start training |
| GET | `/api/train/runs` | List runs |
| GET | `/api/train/runs/{id}` | Status + stats |
| GET | `/api/train/runs/{id}/events` | SSE: step, loss, validation paths (mirror `/api/runs/{id}/events`) |
| POST | `/api/train/runs/{id}/cancel` | Cooperative cancel |
| GET | `/api/train/runs/{id}/artifacts/{path}` | Validation MP4s, final LoRA |
| POST | `/api/train/runs/{id}/register-lora` | Copy LoRA into `web_outputs/loras` + preset entry |

**Worker model:**

- Separate `TrainingWorker` thread pool (`max_workers=1`), analogous to generation executor.
- **Global mutex** with generation: starting train rejects if `server` busy generating (and vice versa) — surface clear UI message.
- Long steps run in `asyncio.to_thread()` / dedicated thread; `step_callback` pushes to asyncio queue for SSE.

**Config builder:**

- Start from upstream YAML templates embedded or loaded from repo `train_configs/`.
- Override: `model.model_path` ← resolved from Web UI active MLX model snapshot.
- Override: `data.preprocessed_data_root`, `output_dir`, `optimization.steps`, `lora.rank`, validation prompts.
- Validate via `LtxTrainerConfig.model_validate()` before start.

### Dependency update

Add commented optional install in `requirements.txt`:

```bash
"ltx-trainer-mlx @ git+https://github.com/dgrauet/ltx-2-mlx.git@v0.14.12#subdirectory=packages/ltx-trainer"
```

Pin tag to `LTX2_MLX_GIT_TAG` in `ltx_mlx_backend.py`.

---

## Implementation phases

### Phase 1 — Foundation (MVP)

- [ ] `training` branch: router + empty `/train` shell + nav
- [ ] `ltx_train_backend.py`: healthcheck, config templates, subprocess/thread wrapper for **preprocess only**
- [ ] Dataset upload API + folder layout
- [ ] Train page: upload videos + captions → preprocess button → log/progress panel
- [ ] Docs in README: optional trainer install

**Exit criteria:** User can preprocess clips from Web UI; no training yet.

### Phase 2 — T2V LoRA training

- [ ] `POST /api/train/runs` wrapping `LtxvTrainer.train(step_callback=...)`
- [ ] SSE progress (step / total / ETA from `TrainingStats`)
- [ ] Cancel flag checked between steps
- [ ] Validation MP4 serving from `output_dir`
- [ ] **Register LoRA** → existing `/api/loras/custom` flow

**Exit criteria:** End-to-end T2V LoRA on a toy dataset (≥2 clips); use in generator.

### Phase 3 — Slice + AV presets

- [ ] Slice API (ffmpeg dependency check)
- [ ] `with_audio` preprocess toggle
- [ ] Presets: `lora_av_whisper` simplified form (audio-only target modules hidden behind preset)
- [ ] RAM warning banners (`--low-ram` → `enable_gradient_checkpointing`)

### Phase 4 — V2V / IC-LoRA training

- [ ] Reference video upload + reference latent preprocess path
- [ ] `video_to_video` strategy UI
- [ ] Validation with `reference_videos`

### Phase 5 — Polish

- [ ] Resume from checkpoint (`model.load_checkpoint`)
- [ ] W&B optional (`wandb` extra)
- [ ] MCP tool `ltx_train_lora` for agents (optional)

---

## Risks and constraints

| Risk | Mitigation |
|------|------------|
| OOM during train | Default to q8 model path; expose checkpointing; block train if free RAM estimate low |
| Train + generate concurrent | Global `mlx_busy` lock shared with `LocalVideoGenerator` |
| Preprocess partial HF download | Use same resolved `model_path` as inference (full snapshot already cached) |
| V2V reference latents | Defer to Phase 4; document manual preprocess steps until automated |
| Long jobs lost on server restart | Persist `status.json`; optional resume; warn user |
| Alpha trainer API | Pin v0.14.12; thin adapter layer in `ltx_train_backend.py` |

---

## Testing strategy

1. **Unit** (no MLX): config builder, path layout, `_sync` job state, YAML merge.
2. **Integration** (MLX machine): preprocess 2 clips → 50-step T2V LoRA → load in generator.
3. **Manual**: `/train` SSE progress, cancel mid-run, register LoRA preset.

Upstream tests to mirror behavior: `tests/test_trainer_core.py`, `tests/test_trainer_datasets.py`.

---

## Open questions (decide before Phase 2)

1. **Default base weights for training** — always `transformer-dev.safetensors` (inference-compatible CFG pipelines) vs distilled (faster but different inference path)?
2. **Dataset size limits** — cap uploads (e.g. 2 GB / 50 clips) for Web UI?
3. **Separate process** — run trainer in child process for crash isolation vs in-process thread?
4. **Standalone `web_server.py`** — should `/train` work without full `server.py` WS stack? (Recommend: yes, train-only via FastAPI.)

---

## Immediate next step

Implement **Phase 1** on `training` branch: routing, `ltx_train_backend.py` skeleton, preprocess job API, minimal `/train` UI.

---

## Training inputs (what ltx-2-mlx accepts & requires)

Training is three optional/required stages. **Train** only consumes **preprocessed** data; everything before that is dataset prep.

### Stage A — Slice (optional)

**API:** `ltx_trainer_mlx.slice_clips.slice_videos` · **Requires:** `ffmpeg` on PATH

| Input | Required | Notes |
|-------|----------|-------|
| `sources` | yes | One or more video files or directories |
| `out_dir` | yes | Per-source subfolders of clips |
| `interval` | no (default 4s) | Clip length; ignored if `timecodes_file` set |
| `timecodes_file` | no | `start,end` per line |
| `res` | no (default `384x384`) | `WxH`, both **÷32** |
| `fps` | no (default 24) | Output fps |
| `fit` | no | `crop` or `pad` |
| `min_length` | no | Drop clips shorter than N seconds |
| `max_clips` / `sample` | no | Cap + even/sequential sampling |
| `skip_start` / `skip_end` | no | Trim intros/outros |
| `caption_template` | no | Writes identical `.txt` beside each clip |
| `crf` | no | x264 quality |

**Outputs:** `clip_XXX.mp4` + optional `clip_XXX.txt` (caption seed for editing).

---

### Stage B — Preprocess (required before train)

**API:** `ltx_trainer_mlx.preprocess.preprocess_dataset` · **Requires:** local MLX model dir, Gemma (HF id)

| Input | Required | Notes |
|-------|----------|-------|
| `videos_dir` | yes | `.mp4/.mov/.avi/.mkv/.webm`; recursive (slice subfolders OK) |
| `output_dir` | yes | Creates `output_dir/.precomputed/` |
| `model_dir` | yes | **Local path** to MLX snapshot (encoders only; partial HF download OK in v0.14.12) |
| `gemma_model_id` | no | Default `mlx-community/gemma-3-12b-it-4bit` |
| `target_height` / `target_width` | no | **÷32**; default = native per clip |
| `max_frames` | no | Default 97; must be **8k+1** |
| `captions_dir` | no | `.txt` per video stem; else **filename stem** used as prompt |
| `caption_ext` | no | Default `.txt` |
| `with_audio` | no | Adds `audio_latents/`; **required** if training with `generate_audio: true` |
| `frame_rate` | no | Written into latent metadata; default = probed fps |

**Outputs:**

```
<preprocessed>/.precomputed/
  latents/latent_0000.safetensors       # video VAE latent + dims/fps metadata
  conditions/condition_0000.safetensors # Gemma video+audio prompt embeds
  audio_latents/latent_0000.safetensors # optional; same index as video latent
```

**V2V add-on (Phase 4):** IC-LoRA also needs `reference_latents/latent_XXXX.safetensors` paired by index (separate encode pass — not in basic `preprocess_dataset` today; manual or custom script per upstream `lora_v2v.yaml`).

---

### Stage C — Train (LoRA / full)

**API:** `LtxvTrainer(LtxTrainerConfig).train(step_callback=…)` · **Requires:** `ltx-trainer-mlx`, preprocessed data, **local** `model.model_path`

#### Hard requirements (`LtxTrainerConfig` validation)

| Field | Required | Notes |
|-------|----------|-------|
| `model.model_path` | yes | Existing **local** directory (not URL) |
| `model.text_encoder_path` | yes* | Gemma id/path; *skipped if no validation prompts |
| `model.training_mode` | yes | `lora` (UI default) or `full` |
| `lora` block | yes if `lora` mode | `rank`, `alpha`, `dropout`, `target_modules` |
| `data.preprocessed_data_root` | yes | Path to dataset root (parent of `.precomputed`) |
| `training_strategy.name` | yes | `text_to_video` or `video_to_video` |
| `optimization.steps` | yes | Often 1000–3000+ |
| `output_dir` | yes | Checkpoints + validation MP4s |

#### Common optional / preset fields

| Field | Purpose |
|-------|---------|
| `model.transformer_file` | e.g. `transformer-dev.safetensors` (AV/style LoRAs) |
| `model.load_checkpoint` | Resume from prior checkpoint dir/file |
| `training_strategy.generate_audio` | Joint AV training (needs audio latents) |
| `optimization.enable_gradient_checkpointing` | **Required** on 64 GB for dev-base; maps to CLI `--low-ram` |
| `optimization.batch_size`, `gradient_accumulation_steps`, `learning_rate`, schedulers | Standard training knobs |
| `validation.*` | Prompts, `video_dims` (W,H,F), `interval`, `inference_steps`, `reference_videos` (V2V) |
| `checkpoints.interval` / `keep_last_n` | Intermediate `.safetensors` |
| `flow_matching.timestep_sampling_mode` | Default `shifted_logit_normal` |
| `seed` | Reproducibility |
| `wandb.*` / `hub.*` | Off by default in UI |

#### Strategy matrix (what we wire first)

| Preset | `training_strategy` | Preprocess | `transformer_file` | RAM hint |
|--------|---------------------|------------|----------------------|----------|
| **T2V style** | `text_to_video`, `generate_audio: false` | standard | auto (distilled OK) | 32–48 GB |
| **AV style** | `text_to_video`, `generate_audio: true` | `--with-audio` | `transformer-dev.safetensors` | 64 GB + checkpointing |
| **IC-LoRA V2V** | `video_to_video`, LoRA only | + `reference_latents/` | LoRA | defer Phase 4 |

#### Trainer outputs

- `output_dir/checkpoint-XXXX.safetensors` (LoRA weights)
- `output_dir/validation_step_XXXX_*.mp4` (when `validation.interval` set)
- `output_dir/config.yaml` (resolved config copy)
- Final return: `(saved_path: Path, TrainingStats)` — steps/sec, peak GB, total time

#### Trainer progress hooks (for our adapter)

| Hook | Data available |
|------|----------------|
| `step_callback(global_step, total_steps, validation_paths)` | Step index, validation MP4 paths after val steps |
| `TrainingProgress.update_training` | `loss`, `lr`, `step_time` (internal — we patch or subclass to expose) |
| `disable_progress_bars=True` | Logs loss every 5 steps to logger (fallback) |

**No built-in cancel** — cooperative cancel via `step_callback` raising `TrainingCancelledError` between steps.

---

## What we wire in ltx-ws (scope by preset)

### Phase 1–2 UI fields → upstream mapping

| UI control | Maps to |
|------------|---------|
| Upload videos + caption files | `videos_dir` (+ optional `captions_dir`) |
| “Slice first” toggle + interval/res/fps/template | `slice_videos(...)` → `clips/` |
| Model picker | `model_dir` = same resolved snapshot as inference (`state.active_model`) |
| Resolution / max frames / with audio | `preprocess_dataset(...)` |
| Preset: T2V / AV | Load embedded YAML template → override paths & steps |
| Rank, steps, LR, val interval, val prompts | `LtxTrainerConfig` overrides |
| Low RAM toggle | `optimization.enable_gradient_checkpointing=true` |
| Run name | `output_dir` subfolder + preset label |

### Phase 4 additions (V2V)

| UI control | Maps to |
|------------|---------|
| Reference video per target clip | `reference_latents/` preprocess + `validation.reference_videos` |
| `reference_downscale_factor` | validation config |

### Out of scope for v1 UI (CLI / advanced YAML only)

- `training_mode: full` (full fine-tune)
- W&B / Hub push
- Custom `target_modules` (expose in “Advanced YAML” panel later)
- Timecode-list slicing

---

## Long-running background jobs & client updates

Reuse the **generation run pattern** in `web_ui.py` — it already solves queueing, SSE, cancel, and persistence. Training jobs are longer and multi-phase but fit the same model.

### Job model: `TrainJob` (extends run concepts)

One **`job_id`** spans all phases (not three separate IDs):

```text
phase: queued → slicing → preprocessing → training → done | failed | cancelled
```

Persisted to `web_outputs/train/<job_id>/status.json` (+ index in `settings.json`).

```json
{
  "job_id": "...",
  "phase": "training",
  "preset": "t2v",
  "created_at": "...",
  "step": 420,
  "total_steps": 3000,
  "loss": 0.0842,
  "lr": 0.00035,
  "eta_s": 3600,
  "peak_memory_gb": 28.4,
  "validation_clips": [{"step": 400, "url": "/api/train/jobs/.../validation/400_0.mp4"}],
  "artifact_lora": "/api/train/jobs/.../lora.safetensors",
  "error": null
}
```

### Worker architecture

```
┌─────────────────────────────────────────────────────────┐
│  FastAPI (web_ui)                                       │
│  POST /api/train/jobs  → enqueue job_id                 │
│  GET  /api/train/jobs/{id}/events  → SSE (EventSource)  │
└───────────────────────┬─────────────────────────────────┘
                        │
        ┌───────────────▼───────────────┐
        │  _train_worker_loop (async)    │  ← mirror _worker_loop
        │  asyncio.Queue[job_id]           │
        └───────────────┬───────────────┘
                        │ asyncio.to_thread()
        ┌───────────────▼───────────────┐
        │  ltx_train_backend.py          │
        │  · slice_videos (ffmpeg)       │
        │  · preprocess_dataset (MLX)    │
        │  · LtxvTrainer.train (MLX)     │
        └───────────────────────────────┘
```

**MLX exclusivity:** shared `AppState.mlx_busy: asyncio.Lock` — training and generation cannot overlap (same as today’s single gen executor). `POST /api/generate` returns 409 if train active; `POST /api/train/jobs` returns 409 if gen active.

**Threading:** MLX training blocks the GIL/Metal for minutes–hours; run entire `slice` / `preprocess` / `train` in **`asyncio.to_thread()`** (or dedicated `ThreadPoolExecutor(max_workers=1)`), same as LoRA downloads. Main asyncio loop stays responsive for SSE pings.

### SSE event schema (mirror `/api/runs/{id}/events`)

Client uses **`EventSource`** on `/api/train/jobs/{job_id}/events` (same as `subscribeRun` in `App.tsx`). Optional later: WS `train_progress` for raw `server.py` clients.

| Event `type` | When | Payload |
|--------------|------|---------|
| `job_started` | Job dequeued | `job_id`, `preset`, `phases` |
| `phase_started` | slice / preprocess / train begin | `phase`, `message` |
| `phase_progress` | preprocess clip N/M | `phase`, `current`, `total`, `message` |
| `train_step` | each optim step | `step`, `total`, `loss`, `lr`, `step_time_s`, `eta_s`, `peak_memory_gb` |
| `train_validation` | val interval | `step`, `videos: [{url, prompt}]` |
| `train_checkpoint` | checkpoint saved | `step`, `path` |
| `ping` | 120s idle | `{}` |
| `job_done` | success | `artifact_lora`, `stats`, `register_lora_url` |
| `error` | failure | `message`, `phase` |
| `job_complete` | always (finally) | `job_id`, `status` |

**Loss streaming:** wrap `TrainingProgress.update_training` in `ltx_train_backend.py` to push `loss`/`lr` into a thread-safe queue drained by the training thread’s `step_callback`. Avoid duplicating the 200-line train loop.

**Cancel:** `POST /api/train/jobs/{id}/cancel` sets `job.cancelled=True`; `step_callback` checks flag and raises `TrainingCancelledError` → `phase: cancelled`, emit `job_complete`.

**Reconnect:** SSE handler replays `status.json` snapshot then attaches to live queue (same pattern as completed runs in `run_events`).

### Frontend (`/train`)

- `subscribeTrainJob(jobId)` — clone of `subscribeRun` with `train_step` / `train_validation` handlers
- Progress bar: reuse `formatProgressMessage` / `formatMmSs` from `progress.ts`
- Phase stepper: Slice → Preprocess → Train
- Validation gallery: thumbnails from `train_validation` events
- **Background-friendly:** user can navigate away; job continues; reconnect via job list + SSE
- Header badge: “Training step 420/3000” when job active (poll `/api/train/jobs/active` or keep SSE open globally)

### WebSocket (optional Phase 2b)

For `server.py` WS clients (videofentanyl), add message types parallel to generation:

- `train_job_status` — polled or pushed during training
- Not required for Web UI (SSE is enough and already works through Vite proxy)

---

## Minimal API surface (revised)

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/train/health` | `ltx_trainer_mlx` installed, ffmpeg, model path resolved |
| GET | `/api/train/presets` | T2V / AV templates + field metadata |
| POST | `/api/train/jobs` | Create job: uploads refs OR multipart in same request |
| GET | `/api/train/jobs` | List jobs (active + history) |
| GET | `/api/train/jobs/{id}` | `status.json` snapshot |
| GET | `/api/train/jobs/{id}/events` | **SSE** progress stream |
| POST | `/api/train/jobs/{id}/cancel` | Cooperative cancel |
| GET | `/api/train/jobs/{id}/artifacts/{name}` | LoRA, validation MP4s |
| POST | `/api/train/jobs/{id}/register-lora` | → existing custom LoRA preset |

Single **`POST /api/train/jobs`** body (Phase 2):

```json
{
  "preset": "t2v",
  "name": "my_style_lora",
  "slice": { "enabled": false },
  "preprocess": { "width": 704, "height": 480, "max_frames": 97, "with_audio": false, "frame_rate": 24 },
  "train": { "steps": 2000, "rank": 64, "learning_rate": 5e-4, "validation_prompts": ["..."], "validation_interval": 500, "checkpoint_interval": 500, "low_ram": false },
  "video_paths": ["uploaded-id-1", "uploaded-id-2"],
  "caption_paths": ["uploaded-id-1.txt"]
}
```

---

## Revised implementation phases

### Phase 1 — Job shell + preprocess
- `TrainJob` + worker queue + SSE skeleton
- Multipart upload → `web_outputs/train/<job_id>/raw/`
- Preprocess phase only; `phase_progress` events

### Phase 2 — T2V training end-to-end
- `LtxvTrainer` wrapper + `train_step` / `train_validation` SSE
- Cancel + `status.json` persistence
- Register LoRA → inference presets
- MLX lock vs generation

### Phase 3 — Slice + AV preset
- Slice phase in job pipeline
- `with_audio` + `lora_av` simplified preset

### Phase 4 — V2V
- Reference latent preprocess + validation reference videos

