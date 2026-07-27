"""IC-LoRA V2V + optional I2V conditioning composition."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from ltx_mlx_backend import (
    IC_LORA_IMAGE_CRF,
    _build_ic_lora_image_conditionings,
    _ic_lora_uses_hdr_pipeline,
    _needs_pose_control_preprocessing,
    _prepare_ic_lora_video_conditioning,
)


def test_build_ic_lora_image_conditionings_frame_zero_only():
    images = _build_ic_lora_image_conditionings("/tmp/char.jpg", 97)
    assert images == [("/tmp/char.jpg", 0, 1.0, IC_LORA_IMAGE_CRF)]


def test_needs_pose_control_for_union_primary():
    with patch("ltx_mlx_backend._ic_lora_reference_downscale_factor", return_value=2):
        with patch("ltx_mlx_backend._ic_lora_uses_hdr_pipeline", return_value=False):
            assert _needs_pose_control_preprocessing(
                [("/loras/union.safetensors", 1.0)], [("m.mp4", 1.0)]
            )


def test_needs_pose_control_false_for_hdr_primary():
    with patch("ltx_mlx_backend._ic_lora_uses_hdr_pipeline", return_value=True):
        assert not _needs_pose_control_preprocessing(
            [("/loras/hdr.safetensors", 1.0)], [("m.mp4", 1.0)]
        )


def test_hdr_detection_uses_primary_only():
    assert not _ic_lora_uses_hdr_pipeline(
        [
            ("/loras/union-control.safetensors", 1.0),
            ("/loras/ic-lora-hdr-0.9.safetensors", 1.0),
        ]
    )
    assert _ic_lora_uses_hdr_pipeline([("/loras/ic-lora-hdr-0.9.safetensors", 1.0)])


def test_prepare_ic_lora_video_conditioning_passthrough_hdr(tmp_path: Path):
    motion = tmp_path / "motion.mp4"
    motion.write_bytes(b"x")
    with patch("ltx_mlx_backend._ic_lora_uses_hdr_pipeline", return_value=True):
        vc, cleanup = _prepare_ic_lora_video_conditioning(
            [(str(motion), 0.9)],
            resolved_loras=[("/loras/hdr.safetensors", 1.0)],
            width=512,
            height=288,
            num_frames=25,
            fps=24.0,
            tmpdir=str(tmp_path),
        )
    assert vc == [(str(motion), 0.9)]
    assert cleanup == []


def test_prepare_ic_lora_video_conditioning_empty_t2v(tmp_path: Path):
    """HDR pure T2V: empty video_conditioning is valid (matches upstream hdr-ic-lora)."""
    with patch("ltx_mlx_backend._ic_lora_uses_hdr_pipeline", return_value=True):
        vc, cleanup = _prepare_ic_lora_video_conditioning(
            [],
            resolved_loras=[("/loras/ic-lora-hdr-0.9.safetensors", 1.0)],
            width=512,
            height=288,
            num_frames=25,
            fps=24.0,
            tmpdir=str(tmp_path),
        )
    assert vc == []
    assert cleanup == []


def test_run_ic_lora_generation_hdr_t2v_allows_empty_vcond(tmp_path: Path):
    """HDR path must not require a reference video."""
    from unittest.mock import MagicMock

    from ltx_mlx_backend import GenerationRequest, _run_ic_lora_generation

    out = tmp_path / "out.mp4"
    req = GenerationRequest(
        prompt="hdr sunset",
        mode="ic_lora",
        skip_stage_2=True,
        reference_strength=0.8,
    )
    gen = MagicMock()
    gen.fps = 24.0
    gen.spill_dir = None
    pipe = MagicMock()
    gen._get_pipe.return_value = pipe
    captured: dict = {}

    def _capture(p, **kwargs):
        captured.update(kwargs)
        captured["_pipe"] = p

    with patch("ltx_mlx_backend._ic_lora_uses_hdr_pipeline", return_value=True):
        with patch("ltx_mlx_backend._tune_ic_lora_strengths", side_effect=lambda x: x):
            with patch(
                "ltx_mlx_backend._prepare_ic_lora_video_conditioning",
                return_value=([], []),
            ):
                with patch(
                    "ltx_mlx_backend._invoke_generate_and_save",
                    side_effect=_capture,
                ):
                    _run_ic_lora_generation(
                        gen,
                        req=req,
                        prompt="hdr sunset",
                        resolved_loras=[("/loras/ic-lora-hdr-0.9.safetensors", 1.0)],
                        vc_items=[],
                        tmp_image=None,
                        tmpdir=str(tmp_path),
                        out_path=str(out),
                        width=512,
                        height=288,
                        nf=25,
                        seed=1,
                        steps=8,
                        tmp_video_conditioning_cleanup=[],
                    )

    gen._get_pipe.assert_called_once()
    assert gen._get_pipe.call_args[0][0] == "hdr_ic_lora"
    assert captured["video_conditioning"] == []
    assert captured["skip_stage_2"] is True
    assert captured["conditioning_attention_strength"] == pytest.approx(0.8)
    assert "images" not in captured


def test_run_ic_lora_generation_hdr_i2v_passes_image(tmp_path: Path):
    from unittest.mock import MagicMock

    from ltx_mlx_backend import GenerationRequest, _run_ic_lora_generation

    img = tmp_path / "start.jpg"
    img.write_bytes(b"x")
    req = GenerationRequest(prompt="i2v", mode="ic_lora")
    gen = MagicMock()
    gen.fps = 24.0
    gen.spill_dir = None
    captured: dict = {}

    with patch("ltx_mlx_backend._ic_lora_uses_hdr_pipeline", return_value=True):
        with patch("ltx_mlx_backend._tune_ic_lora_strengths", side_effect=lambda x: x):
            with patch(
                "ltx_mlx_backend._prepare_ic_lora_video_conditioning",
                return_value=([], []),
            ):
                with patch(
                    "ltx_mlx_backend._invoke_generate_and_save",
                    side_effect=lambda p, **kw: captured.update(kw),
                ):
                    _run_ic_lora_generation(
                        gen,
                        req=req,
                        prompt="i2v",
                        resolved_loras=[("/loras/ic-lora-hdr-0.9.safetensors", 1.0)],
                        vc_items=[],
                        tmp_image=str(img),
                        tmpdir=str(tmp_path),
                        out_path=str(tmp_path / "o.mp4"),
                        width=512,
                        height=288,
                        nf=25,
                        seed=1,
                        steps=8,
                        tmp_video_conditioning_cleanup=[],
                    )

    assert captured["images"] == [(str(img), 0, 1.0, IC_LORA_IMAGE_CRF)]
