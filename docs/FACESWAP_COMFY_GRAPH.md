# BFS V3 Comfy workflow graph (extracted)

Source: `Alissonerdx/BFS-Best-Face-Swap-Video` → `workflows/workflow_ltx2_head_swap_drag_and_drop_v3.0.json`

**Active path (V3):** single-stage distilled sampler at **full canvas resolution**. No half-res stage, no latent upscale in the wired graph (upscale model is loaded but not consumed).

## Generation parameters (from JSON literals)

| Parameter | Value | Node |
|-----------|-------|------|
| Base resolution (longest edge) | 1024 | `PrimitiveInt` 362 |
| FPS | 24 | `PrimitiveFloat` 358 → `Set_FPS` |
| Duration | 5 s | `PrimitiveFloat` 427 |
| Total frames | `((fps * duration) // 8) * 8 + 1` → 121 @ 24×5 | `SimpleCalculatorKJ` 352 |
| Sampler | `euler_ancestral_cfg_pp` | `KSamplerSelect` 386 |
| Scheduler | `bong_tangent`, steps=8, denoise=1 | `BasicScheduler` 575 |
| Sigmas (manual table, orphaned) | `1.0, 0.99375, …, 0.0` (9 values = 8 steps) | `ManualSigmas` 391 — **not wired**; matches MLX `DISTILLED_SIGMAS` |
| CFG | 1.0 | `CFGGuider` 396 |
| Frame rate conditioning | 24 | `LTXVConditioning` 390 |

## LoRA stack (order matters)

Applied on **distilled** UNet (`UNETLoader` 478 → `ltx-2.3-22b-distilled_transformer_only_fp8_input_scaled_v3.safetensors`):

1. `ltx-2/ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors` @ **1.0** — `LoraLoaderModelOnly` 419
2. `ltx-2/2.3/v3/lora_weights_step_10000.safetensors` @ **0.8** — `LoraLoaderModelOnly` 537
3. `ltx-2/2.3/head_swap_extracted_mix_rank_adaptive_fro_0.98_fp16_00001_.safetensors` @ **1.0** — `LoraLoaderModelOnly` 573

MLX port: fuse **distilled dynamic LoRA** from model dir (if present) + user **head-swap** LoRA. Optional BFS training weights LoRA is not required in model bundle.

## Guide / composite

| Step | Node | Parameters |
|------|------|------------|
| Composite | `ReservedRegionFrameComposer` 360 | `left`, region **256 px**, `all_faces_every_frame`, face scale 100, chroma **(0,255,0)**, loop |
| Resize | `ImageResizeKJv2` 361 | 1024×1024 lanczos, multiple-of-32 |
| Add guide | `LTXVAddGuideMulti` 440 | **1** guide, `frame_idx=0`, `strength=1.0` |
| VAE encode guide | `VAEEncode` 443 | composite `control_video` |
| Crop after sample | `LTXVCropGuides` 383 | strips appended guide latents |

**Not in graph:** `LTXVPreprocess` / CRF node — BFS README recommends CRF ~29–33; MLX applies `LTXVPreprocess` in `encode_guide_video` (default CRF 33).

**Not used:** IC-LoRA reference append, LipDub, `normalize_video_for_ic_lora_reference`.

## Conditioning

- Positive: `Get_positive_conditioning` ← optional `TextGenerateLTX2Prompt` when switch `Automatic Prompt?` = true (default **on**)
- Negative: fixed quality guard string on `CLIPTextEncode` 395 (`pc game, cartoon, …`)
- At CFG=1.0, guider is effectively single-pass (MLX: `denoise_loop`, positive-only text encode)

## Logical MLX mapping

```text
reference + face → ReservedRegionFrameComposer → guide MP4
  → LTXVPreprocess (CRF) → VAE encode → append_keyframe (frame_idx=0, strength=1)
  → distilled DiT + fused LoRAs → denoise 8 steps (DISTILLED_SIGMAS) @ full res
  → LTXVCropGuides → VAE decode → crop main panel
```

## Out of scope (present in repo, not V3 path)

- `LatentUpscaleModelLoader` 418 — set but never read by sampler
- `ManualSigmas` 391 — disconnected
- Wan22 / Bernini / v1–v2 workflow files
