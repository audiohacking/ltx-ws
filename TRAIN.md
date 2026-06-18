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
