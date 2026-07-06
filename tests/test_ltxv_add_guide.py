"""Tests for Comfy LTXVAddGuide local primitives."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_ltxv_add_guide import (
    VideoConditionByAppendedGuide,
    compute_guide_video_positions,
    crop_guides_from_video_tokens,
    generation_token_count,
    ltxv_preprocess_rgb_frame,
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
    out = cond.apply(state, spatial_dims=(2, 2, 3))

    assert out.latent.shape[1] == gen_n + guide_n
    assert float(mx.sum(out.latent[:, gen_n:, :]).item()) == 0.0
    assert float(mx.mean(out.clean_latent[:, gen_n:, :]).item()) == 1.0
    assert float(mx.mean(out.denoise_mask[:, gen_n:, :]).item()) == 0.0
    assert float(mx.mean(out.denoise_mask[:, :gen_n, :]).item()) == 1.0


def test_ltxv_preprocess_passthrough_when_crf_zero():
    pytest.importorskip("av")
    frame = np.ones((64, 64, 3), dtype=np.float32) * 0.5
    out = ltxv_preprocess_rgb_frame(frame, crf=0)
    assert out.shape == frame.shape
    assert np.allclose(out, frame)


def test_ltxv_preprocess_changes_frame_when_crf_positive():
    pytest.importorskip("av")
    rng = np.random.default_rng(0)
    frame = rng.random((64, 64, 3), dtype=np.float32)
    out = ltxv_preprocess_rgb_frame(frame, crf=33)
    assert out.shape[0] >= 62 and out.shape[1] >= 62
    assert not np.allclose(out[:62, :62], frame[:62, :62], atol=0.02)


def test_crop_guides_from_video_tokens():
    tokens = mx.zeros((1, 20, 128))
    cropped = crop_guides_from_video_tokens(tokens, num_generation_tokens=12)
    assert cropped.shape == (1, 12, 128)


def test_ltxv_preprocess_accepts_bfloat16_video_tensor():
    pytest.importorskip("av")
    import mlx.core as mx

    from ltx_ltxv_add_guide import preprocess_guide_video_tensor

    video = mx.ones((1, 3, 2, 64, 64), dtype=mx.bfloat16) * 0.5
    out = preprocess_guide_video_tensor(video, crf=0)
    assert out.dtype == mx.float32
    out2 = preprocess_guide_video_tensor(video, crf=33)
    assert out2.dtype == mx.float32
    assert out2.shape == video.shape


def test_compute_guide_positions_match_generation_at_frame_zero():
    import numpy as np

    from ltx_core_mlx.utils.positions import compute_video_positions
    from ltx_ltxv_add_guide import compute_guide_video_positions

    for f, h, w in ((4, 8, 10), (13, 6, 9)):
        ref = np.array(compute_video_positions(f, h, w, frame_rate=24.0))
        guide = np.array(compute_guide_video_positions(f, h, w, frame_rate=24.0, frame_idx=0))
        assert guide.shape == ref.shape
        np.testing.assert_allclose(guide, ref, rtol=1e-5, atol=1e-5)


def test_face_swap_pipeline_uses_add_guide_in_source():
    from pathlib import Path

    pipe = Path("ltx_face_swap_pipeline.py").read_text(encoding="utf-8")
    guide = Path("ltx_ltxv_add_guide.py").read_text(encoding="utf-8")
    assert "ltx_ltxv_add_guide" in pipe
    assert "encode_guide_video" in pipe
    assert "ltxv_preprocess_rgb_frame" in guide
    assert "crop_guides_from_video_tokens" in pipe
    assert "denoise_loop" in pipe
    assert "DISTILLED_SIGMAS" in pipe
    assert "guided_denoise_loop" not in pipe
    assert "self.upsampler" not in pipe
    assert "extract_bfs_guide_keyframe_images" not in pipe
    assert "append_ic_lora_reference_video_conditionings" not in pipe


def test_face_swap_pipeline_class_exports():
    from ltx_face_swap_pipeline import (
        DEFAULT_FACE_SWAP_CFG,
        DEFAULT_FACE_SWAP_NUM_STEPS,
        FaceSwapPipeline,
    )
    from ltx_pipelines_mlx._base import BasePipeline

    assert issubclass(FaceSwapPipeline, BasePipeline)
    assert hasattr(FaceSwapPipeline, "generate_face_swap")
    assert DEFAULT_FACE_SWAP_CFG == 1.0
    assert DEFAULT_FACE_SWAP_NUM_STEPS == 8
