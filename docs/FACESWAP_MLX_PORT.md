# BFS Face Swap — MLX Port (Complete)

**Status:** Implemented on `main` / `faceswap` (local Comfy primitives in `ltx_ltxv_add_guide.py`).

Canonical Comfy reference: [RunComfy LTX 2.3 Face Swap](https://www.runcomfy.com/comfyui-workflows/ltx-2-3-video-face-swap-in-comfyui-realistic-face-replacement-workflow)  
LoRA: [BFS Best Face Swap Video V3](https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap-Video)

---

## Implemented stack

| Comfy node / step | Local module |
|-------------------|--------------|
| `ReservedRegionFrameComposer` | `ltx_face_swap_compose.compose_bfs_v3_guide_video` |
| `LTXVPreprocess` (CRF) | `ltx_ltxv_add_guide.ltxv_preprocess_rgb_frame` |
| `LTXVAddGuide` / `append_keyframe` | `VideoConditionByAppendedGuide` + `encode_guide_video` |
| `LTXVCropGuides` | `crop_guides_from_video_tokens` |
| Dev + CFG sampler (~20 steps) | `FaceSwapPipeline` stage 1 (`guided_denoise_loop`) |
| Head-swap LoRA | `ICLoraPipeline._fuse_loras` on dev transformer |
| Latent 2× upscale + stage 2 | `KeyframeInterpolationPipeline` upsampler path |
| Main-panel crop | `crop_face_swap_output_to_main_video` |

**Not used (wrong for BFS):** LipDub, IC-LoRA reference append, sparse keyframe interpolation.

---

## Usage

**Web UI:** mode **Face swap (LTX 2.3)** — face image + reference video + head-swap LoRA preset.

**MCP / API:** `mode: face_swap`, `image` (identity), `video` (performance), one `lora_specs` entry.

**Weights:** `dgrauet/ltx-2.3-mlx` or `dgrauet/ltx-2.3-mlx-q8` (dev transformer required).

---

## Acceptance checklist (remote test)

1. Spill `*_face_swap_guide.mp4` — green strip + identity on every frame
2. Logs: `LTXVAddGuide`, `crf=33`, `crop_guides_after=yes`, no `ic_ref_append`
3. Identity from reference image; motion from source video
4. Output framed to main panel (no green strip)

---

## Files

- `ltx_ltxv_add_guide.py` — Comfy guide primitives (local patch)
- `ltx_face_swap_compose.py` — BFS composite guide builder
- `ltx_face_swap_pipeline.py` — two-stage dev+CFG pipeline
- `tests/test_ltxv_add_guide.py`, `tests/test_face_swap_*.py`

Future upstream: optional contribution of `ltx_ltxv_add_guide` to `ltx-2-mlx` after validation.
