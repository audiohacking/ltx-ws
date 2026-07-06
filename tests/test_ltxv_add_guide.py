"""Tests for Comfy LTXVAddGuide local primitives."""

from __future__ import annotations

import mlx.core as mx

from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_ltxv_add_guide import (
    VideoConditionByAppendedGuide,
    compute_guide_video_positions,
    crop_guides_from_video_tokens,
    generation_token_count,
    vae_compatible_frame_count,
)


def test_vae_compatible_frame_count():
    assert vae_compatible_frame_count(121, 889) == 121
    assert vae_compatible_frame_count(25, 889) == 25
    assert vae_compatible_frame_count(121, 2) == 9


def test_generation_token_count_aligns_with_patchifier():
    f, h, w, n = generation_token_count(97, 576, 832)
    assert f >= 1 and h >= 1 and w >= 1
    assert n == f * h * w


def test_compute_guide_positions_shape():
    pos = compute_guide_video_positions(4, 8, 10, frame_rate=24.0, frame_idx=0)
    assert pos.shape == (1, 4 * 8 * 10, 3)


def test_appended_guide_extends_sequence_and_freezes_tokens():
    gen_n = 6
    guide_n = 4
    c = 8
    state = LatentState(
        latent=mx.zeros((1, gen_n, c)),
        clean_latent=mx.zeros((1, gen_n, c)),
        denoise_mask=mx.ones((1, gen_n, 1)),
        positions=mx.zeros((1, gen_n, 3)),
    )
    guide_tokens = mx.ones((1, guide_n, c))
    guide_pos = mx.zeros((1, guide_n, 3))
    cond = VideoConditionByAppendedGuide(guide_tokens, guide_pos, strength=1.0)
    out = cond.apply(state, spatial_dims=(2, 2, 3))  # 2*2*3=12 != gen_n — num_noisy uses spatial_dims

    assert out.latent.shape[1] == gen_n + guide_n
    assert float(mx.mean(out.denoise_mask[:, gen_n:, :]).item()) == 0.0
    assert float(mx.mean(out.denoise_mask[:, :gen_n, :]).item()) == 1.0


def test_crop_guides_from_video_tokens():
    tokens = mx.zeros((1, 20, 128))
    cropped = crop_guides_from_video_tokens(tokens, num_generation_tokens=12)
    assert cropped.shape == (1, 12, 128)


def test_face_swap_pipeline_uses_add_guide_in_source():
    from pathlib import Path

    src = Path("ltx_face_swap_pipeline.py").read_text(encoding="utf-8")
    assert "ltx_ltxv_add_guide" in src
    assert "encode_guide_video" in src
    assert "crop_guides_from_video_tokens" in src
    assert "extract_bfs_guide_keyframe_images" not in src
    assert "append_ic_lora_reference_video_conditionings" not in src


def test_face_swap_pipeline_class_exports():
    from ltx_face_swap_pipeline import (
        DEFAULT_FACE_SWAP_CFG,
        DEFAULT_FACE_SWAP_STAGE1_STEPS,
        FaceSwapPipeline,
    )
    from ltx_pipelines_mlx.keyframe_interpolation import KeyframeInterpolationPipeline

    assert issubclass(FaceSwapPipeline, KeyframeInterpolationPipeline)
    assert hasattr(FaceSwapPipeline, "generate_face_swap")
    assert DEFAULT_FACE_SWAP_CFG == 3.0
    assert DEFAULT_FACE_SWAP_STAGE1_STEPS == 20
