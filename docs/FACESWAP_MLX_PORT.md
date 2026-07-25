# BFS Face Swap — MLX Port Plan

**Branch:** `faceswap` only (do not merge to `main` until Phase 6 exit criteria pass)  
**Goal:** Reproduce the [RunComfy BFS V3 workflow](https://www.runcomfy.com/comfyui-workflows/ltx-2-3-video-face-swap-in-comfyui-realistic-face-replacement-workflow) on MLX with **graph isomorphism**, not approximate reuse of IC-LoRA / keyframe / LipDub paths.

**Ground truth (read before changing code):**

| Source | What to extract |
|--------|-----------------|
| [ComfyUI `nodes_lt.py`](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy_extras/nodes_lt.py) | `LTXVPreprocess`, `LTXVAddGuide`, `LTXVCropGuides`, `append_keyframe`, `encode`, `get_latent_index` |
| [Lightricks `keyframe_cond.py`](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-core/src/ltx_core/conditioning/types/keyframe_cond.py) | `VideoConditionByKeyframeIndex` positions + append semantics |
| [ComfyUI-BFSNodes `nodes.py`](https://github.com/alisson-anjos/ComfyUI-BFSNodes/blob/main/nodes.py) | `ReservedRegionFrameComposer` parameters |
| [BFS HF README](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap-Video) | V3 composite workflow, prompt template, LoRA strength |
| HF `workflows/workflow_ltx2_head_swap_drag_and_drop_v3.0` | **Exact** node graph, step counts, CFG, resolution, LoRA order |
| MLX `ltx_pipelines_mlx` | `guided_denoise_loop`, `create_noised_state`, `KeyframeInterpolationPipeline` upscale path |

Current code (`ltx_ltxv_add_guide.py`, `FaceSwapPipeline`) is **Phase 2–3 draft**. Gaps are **scheduled work**, not acceptable drift.

---

## Target graph (Comfy logical order)

```text
reference video + identity image
  → ReservedRegionFrameComposer          # every-frame composite (green strip + face + performance)
  → LTXVPreprocess (CRF per frame)       # before VAE encode
  → empty / img2vid latent @ canvas size
  → LTXVAddGuide (full video, frame_idx=0, strength=1.0)
  → LoraLoaderModelOnly (head_swap @ 0.98)   # NOT IC-LoRA reference append
  → LTXVConditioning (frame_rate)
  → KSampler dev (~20 steps, CFG ~3, STG per LTX_2_3_PARAMS)
  → LTXVCropGuides
  → latent 2× upscale
  → KSampler pass 2 (distilled LoRA; params from workflow JSON)
  → LTXVCropGuides (if workflow applies guide again)
  → VAE decode
  → spatial crop to main panel (strip removed)
```

MLX mapping:

| Comfy | `faceswap` module (target) |
|-------|---------------------------|
| Composer | `ltx_face_swap_compose.compose_bfs_v3_guide_video` |
| Preprocess + AddGuide + Crop | `ltx_ltxv_add_guide.py` |
| Sampler graph | `ltx_face_swap_pipeline.FaceSwapPipeline` |
| API / UI | `ltx_mlx_backend.py`, `web_ui.py`, `mcp_server.py` |

---

## Phase 0 — Workflow extraction (blocking)

**Owner deliverable:** `docs/FACESWAP_COMFY_GRAPH.md` — one row per workflow node.

1. Download `workflow_ltx2_head_swap_drag_and_drop_v3.0` from [BFS HF repo](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap-Video).
2. For each node record: `class_type`, inputs (literal values), wire sources, output slot used downstream.
3. Resolve **from JSON**, not inference:
   - Stage 1 pixel resolution (full canvas vs half-res)
   - `steps`, `cfg`, `stg`, scheduler / sigma source
   - Stage 2 steps and whether CFG runs again
   - LoRA load order: head_swap before/after distilled; stage-2 re-apply?
   - `crf` / `img_compression` on AddGuideAdvanced
   - `blur_radius`, `interpolation`, `crop` on guide preprocess
4. Mark which nodes are **in** vs **out** of the published V3 path (ignore LipDub / Bernini / EditAnything nodes).

**Exit:** Graph doc reviewed; no open “we think Comfy does X” items.

---

## Phase 1 — Guide preparation isomorphism

**Comfy ref:** `ReservedRegionFrameComposer` + video load/resize before AddGuide.

| Task | Action | Comfy parameter | Current `faceswap` |
|------|--------|---------------|-------------------|
| 1.1 | Match composer defaults | `region_size_px=256`, green chroma, `all_faces_every_frame` / identity every frame | Implemented — verify with side-by-side frame dump |
| 1.2 | **Remove** IC-LoRA guide normalize | N/A in Comfy AddGuide path | **Remove** `normalize_video_for_ic_lora_reference` from `_prepare_face_swap_guide_video`; pad to `8k+1` in compose only |
| 1.3 | Canvas sizing | Output frame size = source aspect, performance fit in main panel | `resolve_face_swap_canvas_size` — add regression PNG test vs Comfy export |
| 1.4 | Frame count | `8k+1` before VAE | Single function: `vae_compatible_frame_count` used in compose + encode |

**Exit:** Spill `*_face_swap_guide.mp4` byte-matched in layout (strip, face position, main panel aspect) against Comfy composite on same inputs.

---

## Phase 2 — `ltx_ltxv_add_guide.py` line port (`nodes_lt.py`)

Port **functions**, not behavior labels. Each sub-phase has a golden test.

### 2.1 `LTXVPreprocess`

- Port `preprocess()`, `encode_single_frame()`, `decode_single_frame()` from `nodes_lt.py` (PyAV H.264 in-memory, even H/W).
- Default CRF: **from workflow JSON** (Comfy node default 35; BFS examples use ~29–33 — use workflow value).
- Apply to **every frame** of composite guide before VAE encode.
- Implement `blur_radius` from `LTXVAddGuideAdvanced` when workflow `blur_radius > 0`.

**Test:** `tests/test_ltxv_preprocess_comfy_parity.py` — same random 64×64 frame, MLX vs exported Comfy tensor, max abs diff &lt; ε.

### 2.2 `LTXVAddGuide.encode`

Port `LTXVAddGuide.encode()`:

- Trim pixel frames: `(N-1)//8*8+1`
- Upscale to `latent_w * width_scale`, `latent_h * height_scale` (bilinear, center crop)
- VAE encode → `guiding_latent` `(B,C,T,H,W)`

Implement **causal prepend+strip** when `frame_idx != 0` and multi-frame guide (Comfy lines 438–454) even if BFS uses `frame_idx=0`.

### 2.3 `append_keyframe`

Port exactly:

```python
# upstream torch (keyframe_cond.py):
latent  = cat([state.latent,  zeros_like(tokens)])
clean   = cat([state.clean_latent, tokens])
mask    = 1.0 - strength  # per guide token
```

- `get_latent_index()` — frame_idx rounding (`frame_idx` divisible by 8 when guide length &gt; 1 and frame_idx != 0).
- Concatenate in **latent 5D time axis** then patchify, **or** prove token-append equivalent with golden test (same mask/positions/denoise outcome).

### 2.4 Guide positions (critical)

Do **not** use hand-rolled `compute_video_positions` for appended guides.

Port from `VideoConditionByKeyframeIndex.apply_to`:

1. `patchifier.get_patch_grid_bounds(output_shape=guide_shape)`
2. `get_pixel_coords(latent_coords, scale_factors, causal_fix=...)`
3. `positions[:,0] += frame_idx`; divide temporal axis by `fps`
4. If `num_pixel_frames == 1`, narrow temporal end to `[start, start+1)` in pixel space

Add `compute_guide_positions_from_encoded_latent(encoded_5d, frame_idx, fps, causal_fix)` in `ltx_ltxv_add_guide.py` using MLX patchifier APIs (extend `VideoLatentPatchifier` if grid bounds missing — port from `ltx-core`).

### 2.5 Guide attention metadata

Port `_append_guide_attention_entry` from Comfy/KJNodes if workflow wires `LTXVAddGuideMulti` attention tracking.

Wire into `LatentState.attention_mask` so `guided_denoise_loop` matches Comfy self-attention masking.

### 2.6 `LTXVCropGuides`

Port `LTXVCropGuides.execute`:

- `num_keyframes` from conditioning metadata (track count on append)
- Crop **last** `num_keyframes` latent frames along **time** in 5D, then patchify — or crop last `num_guide_tokens` in token space if proven equivalent
- Clear keyframe metadata after crop

**Phase 2 exit:** `tests/test_ltxv_add_guide_comfy_golden.py` passes for:

- 1-frame guide @ frame 0
- 97-frame composite @ frame 0 @ 480×704 and @ full canvas size from workflow
- round-trip append → crop → shape equals generation latent

---

## Phase 3 — Sampler graph (`FaceSwapPipeline`)

Build pipeline from **Phase 0 graph doc**, not from `KeyframeInterpolationPipeline` convenience.

### 3.1 Model & LoRA order

```text
load dev transformer
  → fuse head_swap LoRA @ 0.98 (Comfy LoraLoaderModelOnly)
  → stage 1 denoise
  → [optional] fuse distilled LoRA for stage 2 (on fused weights — verify Comfy order)
```

Add log lines: `lora_fused=head_swap`, `lora_fused=distilled`, with strengths.

### 3.2 Stage 1 parameters (from workflow JSON)

| Parameter | Source |
|-----------|--------|
| `height`, `width` | Full composite canvas (likely includes strip) |
| `num_frames` | `8k+1` |
| `steps` | workflow KSampler |
| `cfg_scale` | workflow |
| `stg_scale` | LTX_2_3_PARAMS / workflow (likely **1.0**, not 0.0) |
| `rescale_scale`, `modality_scale` | LTX_2_3_PARAMS |
| Sigma schedule | `ltx2_schedule(steps, num_tokens)` or `LTXVScheduler` if workflow uses it |

**Decision gate (Phase 0):** If workflow is **full-res** stage 1, remove half-res stage-1 block; use single canvas resolution end-to-end. If workflow is half-res then upscale, keep two-stage but document node IDs proving it.

### 3.3 Between stages

```text
denoise stage 1
  → LTXVCropGuides
  → VAE denorm → spatial upsampler ×2 → VAE renorm
  → LTXVAddGuide (re-encode composite at full res)
  → stage 2 denoise
  → LTXVCropGuides
  → decode
```

### 3.4 Stage 2

- Steps / CFG / STG from workflow JSON (do not assume `STAGE_2_SIGMAS` length 3 without verification).
- Head-swap LoRA must remain active — verify weights after distilled fuse (log delta norm or compare forward on fixed latent).

### 3.5 Text conditioning

- `format_head_swap_prompt` — shell only; identity from image not text (HF V3).
- Phase 4 adds vision-filled `FACE:` / `ACTION:`.

**Phase 3 exit:** Pipeline logs print explicit graph step names matching Comfy node types; remote run completes without IC-LoRA / keyframe / LipDub code paths.

---

## Phase 4 — Vision prompt (HF V3 optional path)

Port prompt template from BFS README (composite → vision model → structured `head_swap:` block).

- New helper: `ltx_face_swap_prompt.py` — template only, no generation dependency in core path.
- Wire behind API flag `auto_prompt_from_guide=false` default until validated.

**Exit:** With flag on, prompt matches HF format; with flag off, user `ACTION:` only.

---

## Phase 5 — Validation harness

### Automated (CI)

- All Phase 1–2 golden tests
- `tests/test_face_swap_*` compose tests
- Lint / import check `FaceSwapPipeline` registers only on `faceswap` integration tests

### Remote GPU (required before merge)

| # | Check |
|---|--------|
| 1 | Spill guide: green strip + identity every frame |
| 2 | Logs: `LTXVAddGuide`, `LTXVCropGuides`, `crf=`, no `ic_ref_append` |
| 3 | Token counts: `tokens_gen` + `tokens_guide` logged; crop restores `tokens_gen` |
| 4 | Identity: output face ≠ source video face, ≈ reference image |
| 5 | Motion: lips/body track reference (no 8-frame stutter) |
| 6 | Framing: output = main panel crop |
| 7 | Three diverse clips (landscape close-up, dialogue, motion) |

---

## Phase 6 — Merge to `main`

Only when Phase 0–5 exit criteria pass. `main` stays untouched until then.

---

## Work queue (ordered)

| ID | Phase | Task | Files |
|----|-------|------|-------|
| W0 | 0 | Extract workflow JSON → `FACESWAP_COMFY_GRAPH.md` | `docs/` |
| W1 | 1 | Remove IC-LoRA normalize from guide path | `ltx_mlx_backend.py` |
| W2 | 2.4 | Port `get_pixel_coords` guide positions | `ltx_ltxv_add_guide.py`, maybe `patchifiers.py` local |
| W3 | 2.6 | Track `num_keyframes`, crop in 5D or proven token-equiv | `ltx_ltxv_add_guide.py` |
| W4 | 2 | Golden tests vs Comfy export | `tests/` |
| W5 | 0→3 | Set stage-1 resolution + STG/CFG from graph doc | `ltx_face_swap_pipeline.py` |
| W6 | 3 | Align stage-2 params to workflow | `ltx_face_swap_pipeline.py` |
| W7 | 2.5 | Guide attention entry if in graph | `ltx_ltxv_add_guide.py` |
| W8 | 5 | Remote validation matrix | — |
| W9 | 4 | Vision auto-prompt (optional) | `ltx_face_swap_prompt.py` |

**Start now:** W0 → W1 → W2 (blocking for trustworthy test).

---

## Rules of engagement

1. **No new pipeline experiments** without a workflow node ID justification.
2. **Comfy function ports** include a golden or spill-visual test.
3. **`faceswap` branch only** until Phase 6.
4. **No** `Co-authored-by` trailers on commits.
5. Current implementation is **draft** through W5; do not call port complete until W8 passes.
