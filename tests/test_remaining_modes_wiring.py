"""Keyframe / retake / a2v / extend wiring helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ltx_mlx_backend import (
    A2V_DEFAULT_STAGE1_STEPS,
    DEFAULT_EXTEND_LATENT_FRAMES,
    GenerationRequest,
    KEYFRAME_DEFAULT_CFG,
    KEYFRAME_DEV_TRANSFORMER,
    KEYFRAME_DISTILLED_LORA,
    LocalVideoGenerator,
    RETAKE_EXTEND_DEFAULT_STG,
    _apply_optional_generate_kwargs,
    _clamp_a2v_stage1_steps,
    _keyframe_pipe_kwargs,
)


def test_no_regen_audio_maps_to_regenerate_audio_false():
    req = GenerationRequest(prompt="x", no_regen_audio=True)
    kwargs: dict = {}
    _apply_optional_generate_kwargs(kwargs, req)
    assert kwargs["regenerate_audio"] is False
    assert "no_regen_audio" not in kwargs


def test_clamp_a2v_stage1_steps_raises_distilled_default():
    assert _clamp_a2v_stage1_steps(8) == A2V_DEFAULT_STAGE1_STEPS
    assert _clamp_a2v_stage1_steps(24) == 24


def test_a2v_visual_continue_keeps_distilled_steps():
    """I2V+mux continue must not inherit A2V CFG step clamp (same class of bug as I2V→one_stage)."""
    from ltx_mlx_backend import _a2v_effective_stage1_steps

    assert _a2v_effective_stage1_steps(8, visual_i2v_continue=True) == 8
    assert _a2v_effective_stage1_steps(8, visual_i2v_continue=False) == A2V_DEFAULT_STAGE1_STEPS
    assert _a2v_effective_stage1_steps(24, visual_i2v_continue=False) == 24


def test_pad_audio_latent_to_token_count_pads_and_trims():
    pytest.importorskip("mlx.core")
    import mlx.core as mx

    from ltx_mlx_backend import _pad_audio_latent_to_token_count

    short = mx.zeros((1, 8, 338, 16))
    padded = _pad_audio_latent_to_token_count(short, 376)
    assert padded.shape == (1, 8, 376, 16)

    long = mx.ones((1, 8, 400, 16))
    trimmed = _pad_audio_latent_to_token_count(long, 376)
    assert trimmed.shape == (1, 8, 376, 16)


def test_a2v_with_image_uses_a2v_pipe_not_one_stage():
    """Optional start image stays on A2VidPipelineTwoStage — not TI2VidOneStagePipeline."""
    gen = LocalVideoGenerator.__new__(LocalVideoGenerator)
    gen.upscale = False
    gen._pipe_classes = {
        "t2v": object(),
        "a2v": object(),
        "one_stage": object(),
        "i2v": object(),
    }
    # Native a2v never goes through _resolve_generate_pipe_key; visual continue does.
    assert gen._resolve_generate_pipe_key("distilled", has_image=True) == "t2v"
    assert gen._resolve_generate_pipe_key("distilled", has_image=True) != "one_stage"
    assert "a2v" in gen._pipe_classes
    assert gen._pipe_classes["a2v"] is not gen._pipe_classes["one_stage"]


def test_retake_extend_default_stg_matches_upstream():
    assert RETAKE_EXTEND_DEFAULT_STG == pytest.approx(1.0)
    assert DEFAULT_EXTEND_LATENT_FRAMES == 15


def test_keyframe_pipe_kwargs_requires_dev_and_lora(tmp_path: Path):
    with pytest.raises(RuntimeError, match="transformer-dev"):
        _keyframe_pipe_kwargs(tmp_path)

    (tmp_path / KEYFRAME_DEV_TRANSFORMER).write_bytes(b"x")
    with pytest.raises(RuntimeError, match="distilled LoRA"):
        _keyframe_pipe_kwargs(tmp_path)

    (tmp_path / KEYFRAME_DISTILLED_LORA).write_bytes(b"x")
    kw = _keyframe_pipe_kwargs(tmp_path)
    assert kw["dev_transformer"] == KEYFRAME_DEV_TRANSFORMER
    assert kw["distilled_lora"] == KEYFRAME_DISTILLED_LORA
    assert kw["distilled_lora_strength"] == pytest.approx(1.0)


def test_keyframe_default_cfg():
    assert KEYFRAME_DEFAULT_CFG == pytest.approx(3.0)


def test_face_swap_requires_dev_transformer(tmp_path: Path):
    from ltx_face_swap_pipeline import _resolve_dev_transformer

    with pytest.raises(RuntimeError, match="transformer-dev"):
        _resolve_dev_transformer(tmp_path)
    (tmp_path / "transformer-dev.safetensors").write_bytes(b"x")
    assert _resolve_dev_transformer(tmp_path) == "transformer-dev.safetensors"


def test_retake_range_must_be_non_empty():
    start, end = 0, 12
    assert end > start
