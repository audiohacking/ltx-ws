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


def test_prepare_ic_lora_video_conditioning_pose_extract(tmp_path: Path):
    motion = tmp_path / "motion.mp4"
    motion.write_bytes(b"x")
    pose_out = tmp_path / "ic_lora_pose_control.mp4"
    with patch("ltx_mlx_backend._ic_lora_uses_hdr_pipeline", return_value=False):
        with patch("ltx_mlx_backend._ic_lora_reference_downscale_factor", return_value=2):
            with patch("ltx_ic_lora_preprocess.require_pose_control"):
                with patch(
                    "ltx_ic_lora_preprocess.render_pose_control_video",
                    return_value=pose_out,
                ) as render:
                    vc, cleanup = _prepare_ic_lora_video_conditioning(
                        [(str(motion), 0.85)],
                        resolved_loras=[("/loras/union.safetensors", 1.0)],
                        width=64,
                        height=48,
                        num_frames=25,
                        fps=24.0,
                        tmpdir=str(tmp_path),
                    )
    render.assert_called_once()
    assert vc == [(str(pose_out), 0.85)]
    assert cleanup == [str(pose_out)]
