# BFS Face Swap — MLX Port Research Spec

**Status:** Research / not production-ready  
**Branch:** `faceswap` (all prior experiments preserved for reference)  
**Main:** reverted to `af21ef03` (pre–face-swap UI/API)

This document is the canonical plan for porting [BFS Best Face Swap V3](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap-Video) to `ltx-ws` on MLX. **No further pipeline experiments** until the Comfy primitives below are implemented or explicitly mapped to existing MLX APIs.

---

## 1. Why prior `ltx-ws` attempts failed

| Attempt | MLX primitive used | Comfy BFS V3 actually uses | Result |
|--------|-------------------|---------------------------|--------|
| LipDub pipeline | `VideoConditionByReferenceLatent` + audio lock | BFS composite + `LTXVAddGuide` + head-swap LoRA | Original video copied |
| IC-LoRA ref append | IC-LoRA reference tokens @ 1.0 | Guide appended via `append_keyframe`, not IC ref | Original face/motion preserved |
| VAE encode init only | Retake-style encoded latent init | Same init **plus** multi-frame guides + `noise_mask` | Lost motion, no identity |
| Dev + sparse keyframes | `VideoConditionByKeyframeIndex` every 8 frames | Full-sequence `LTXVAddGuide` / batch guides | Choppy video, weak identity |

**Root cause:** We reused IC-LoRA, LipDub, retake, and keyframe-interpolation paths. BFS V3 head-swap LoRA is trained for a **different conditioning contract** — Comfy `LTXVAddGuide` / `LTXVAddGuidesFromBatch` with appended guide latents and `noise_mask`, plus `LTXVCropGuides` after sampling.

---

## 2. What the BFS LoRA expects (authoritative sources)

### 2.1 Model & trigger

- LoRA: `head_swap_v3_rank_adaptive_fro_098.safetensors` @ **0.98** (Comfy workflows; HF notes 1.0 for V1 motion)
- Base: **LTX 2.3** dev UNet path (not distilled-only IC-LoRA)
- Trigger (V3):

```text
head_swap:

FACE:

ACTION:
<performance description — not manual face description if using vision auto-prompt>
```

Identity comes from the **side-panel reference image in the composite guide**, not from describing the face in text ([HF README](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap-Video)).

### 2.2 V3 persistent-template workflow (HF + ComfyUI-BFSNodes)

1. **ReservedRegionFrameComposer** ([source](https://github.com/alisson-anjos/ComfyUI-BFSNodes/blob/main/nodes.py)):
   - Keep output resolution = source video resolution
   - Add **green chroma strip** (default left, ~256px)
   - Place **identity face image in strip on every frame**
   - Fit performance video in remaining panel (no crop of output size)
2. Use composite as **inference guide** (internal only; strip cropped from final output)
3. **Head-swap LoRA** on UNet via standard LoRA loader (`LoraLoaderModelOnly` in Comfy — **not** IC-LoRA reference append)
4. **Dev model + CFG** (~20 steps; CFG ~3; disable distilled “CFG=1” shortcuts for stage 1)
5. Optional: vision model auto-prompt from composite (FACE from strip, ACTION from main panel)

### 2.3 What is *not* BFS V3

- LipDub IC-LoRA
- Union / OpenPose IC-LoRA motion transfer
- Plain `video_conditioning` reference append at strength 1.0
- Sparse keyframe interpolation as a substitute for per-frame guides

---

## 3. Comfy canonical graph (target behavior)

Reference workflow: [RunComfy LTX 2.3 Face Swap](https://www.runcomfy.com/comfyui-workflows/ltx-2-3-video-face-swap-in-comfyui-realistic-face-replacement-workflow)  
HF workflow file: `workflows/workflow_ltx2_head_swap_drag_and_drop_v3.0` (BFS repo)

### 3.1 Node sequence (logical order)

```text
Load video + face image
  → ReservedRegionFrameComposer          # composite guide video (every frame)
  → VAEEncode composite (or empty latent + guides)
  → LTXVAddGuide / LTXVAddGuidesFromBatch  # inject guide latents + noise_mask
  → Load LoRA (head_swap) on MODEL
  → LTXVConditioning + CFG guider
  → KSampler (dev, ~20 steps, cfg≈3)
  → LTXVCropGuides                       # remove appended guide tokens from latent
  → Latent 2× upscale + second sampler pass
  → VAE decode
  → Crop to main panel (no green strip)  # spatial crop, not just latent crop
```

### 3.2 `LTXVAddGuide.append_keyframe` (Comfy core — must port)

Source: [`comfy_extras/nodes_lt.py`](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy_extras/nodes_lt.py) (`LTXVAddGuide`)

For each guide (image or video chunk):

1. **VAE-encode** guide pixels to `guiding_latent` tensor `[B, C, T_guide, H, W]`
2. **`add_keyframe_index`**: patchify guide, compute pixel coords, offset by `frame_idx`, append to conditioning `keyframe_idxs` (RoPE)
3. **`append_keyframe`**:
   - `latent_image = concat(latent_image, guiding_latent)` along **time** dimension
   - `noise_mask = concat(noise_mask, mask)` where `mask = 1.0 - strength` → guide tokens are **frozen** (not denoised)
4. After sampling: **`LTXVCropGuides`** strips last `num_keyframes` latent frames and clears `keyframe_idxs`

This is **not** the same as MLX `VideoConditionByKeyframeIndex` (single-frame conditioning injection without latent concat + noise mask).

### 3.3 BFS V3 guide mode: full video, not sparse PNGs

For persistent template, the guide is a **full composite video**. Comfy options:

- **`LTXVAddGuide`** with multi-frame `image` tensor at `frame_idx=0` (encodes `8k+1` frames, causal first-frame handling)
- **`LTXVAddGuidesFromBatch`** — each batch index `i` with non-black frame becomes guide at frame `i`

**Do not** sample every 8th frame to PNGs unless the Comfy workflow explicitly does so (BFS V3 does not).

### 3.4 Preprocessing on guides

`LTXVAddGuideAdvanced` applies before encode:

- Resize to latent pixel size
- **`LTXVPreprocess(crf)`** — CRF controls motion vs quality tradeoff (default ~29–33)
- Optional blur

KJNodes `LTXVAddGuideMulti` chains multiple `append_keyframe` calls for separate guides at different indices.

---

## 4. MLX gap analysis (`ltx-pipelines-mlx` / `ltx-core-mlx`)

| Comfy primitive | MLX today | Gap |
|----------------|-----------|-----|
| `ReservedRegionFrameComposer` | `ltx_face_swap_compose.py` on `faceswap` branch | **Implemented** — keep, verify against Comfy node |
| `LTXVAddGuide.append_keyframe` | Missing | **Must implement** — latent concat + `noise_mask` + `keyframe_idxs` |
| `LTXVCropGuides` | Missing | **Must implement** — crop appended guide tokens post-sample |
| `LTXVPreprocess(crf)` | Unknown / partial | Verify or port from Comfy |
| Dev + CFG sampler | `RetakePipeline`, `TI2VidTwoStagesPipeline` | **Exists** — reuse `guided_denoise_loop` |
| Head-swap LoRA fuse | `ICLoraPipeline._fuse_loras` | **Exists** — fuse on **dev** transformer before stage 1 |
| IC-LoRA ref append | `append_ic_lora_reference_video_conditionings` | **Do not use** for BFS face swap |
| Keyframe interpolation | `KeyframeInterpolationPipeline` | **Wrong tool** for BFS V3 |
| Spatial output crop | `crop_face_swap_output_to_main_video` | **Implemented** on `faceswap` branch |

---

## 5. Port plan (phased, no improvisation)

### Phase A — Local primitives (`ltx_ltxv_add_guide.py`)

Implemented in-repo (not yet upstreamed):

1. **`encode_guide_video()`** + **`VideoConditionByAppendedGuide`** — Comfy ``append_keyframe``
2. **`crop_guides_from_video_tokens()`** — Comfy ``LTXVCropGuides``
3. **`FaceSwapPipeline`** — full composite guide @ frame 0, dev+CFG stage 1, crop, upscale, stage 2

**Gate:** unit tests in ``tests/test_ltxv_add_guide.py``; remote generation validation on ``faceswap`` branch.

### Phase B — `FaceSwapPipeline` in `ltx-ws`

Only after Phase A:

```text
compose_bfs_v3_guide_video()
  → add_guide (full composite video @ frame_idx=0, strength=1.0, crf=33)
  → fuse head_swap LoRA on dev transformer
  → guided_denoise_loop (cfg=3, steps=20)
  → crop_guides
  → stage-2 upscale + distilled refine (keep head_swap fused through both if Comfy does)
  → decode → crop_face_swap_output_to_main_video()
```

### Phase C — Integration

- Re-enable `face_swap` mode in API / Web UI / MCP **only after** Phase B acceptance tests
- Optional: Ollama/vision auto-prompt from composite (HF template) — not required for identity

---

## 6. Acceptance criteria (before merging to `main`)

1. **Guide video check:** spill `*_face_swap_guide.mp4` shows green strip + identity on every frame; main panel preserves source aspect
2. **Logs:** `add_guide` full composite, `crop_guides` after stage 1 — no `ic_ref_append`, no sparse keyframe extraction
3. **Identity:** output face matches reference image, not source video face
4. **Motion:** lip/expression/body motion tracks source video (no 8-frame stutter)
5. **Framing:** output equals main-panel crop of source (no green strip)
6. **Regression:** `ic_lora`, `lipdub`, `retake`, `extend` unchanged

---

## 7. Branch inventory (`faceswap`)

| Path | Keep / rewrite |
|------|----------------|
| `ltx_face_swap_compose.py` | **Keep** — align tests with Comfy `ReservedRegionFrameComposer` |
| `ltx_ltxv_add_guide.py` | **Keep** — local Comfy AddGuide/CropGuides port |
| `ltx_face_swap_pipeline.py` | **Keep** — uses `ltx_ltxv_add_guide` + dev two-stage |
| `ltx_mlx_backend.py` face_swap block | **Rewrite** — wire new pipeline only |
| `tests/test_face_swap_*.py` | **Rewrite** — test compose + mock Phase A primitives |
| Web UI / MCP face_swap mode | **Do not merge to main** until Phase C |

---

## 8. References

- [BFS Best Face Swap Video (HF)](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap-Video) — V3 workflow description, prompt template
- [ComfyUI-BFSNodes](https://github.com/alisson-anjos/ComfyUI-BFSNodes) — `ReservedRegionFrameComposer`
- [ComfyUI `nodes_lt.py`](https://github.com/comfyanonymous/ComfyUI/blob/master/comfy_extras/nodes_lt.py) — `LTXVAddGuide`, `LTXVCropGuides`, `append_keyframe`
- [ComfyUI-KJNodes `LTXVAddGuideMulti`](https://github.com/kijai/ComfyUI-KJNodes/blob/main/nodes/ltxv_nodes.py) — multi-guide chaining
- [Lightricks ComfyUI-LTXVideo `guide.py`](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/guide.py) — advanced preprocess wrapper
- [RunComfy BFS workflow page](https://www.runcomfy.com/comfyui-workflows/ltx-2-3-video-face-swap-in-comfyui-realistic-face-replacement-workflow)

---

## 9. Next action

1. Open upstream issue/PR in `ltx-2-mlx` for `append_keyframe` + `crop_guides` parity (link this doc)
2. Implement Phase A with Comfy golden tests — **no `ltx-ws` generation experiments until green**
3. Rewrite `FaceSwapPipeline` on this branch per Phase B
