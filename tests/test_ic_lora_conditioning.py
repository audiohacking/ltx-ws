"""IC-LoRA V2V + optional I2V conditioning composition."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from ltx_mlx_backend import (
    IC_LORA_IMAGE_CRF,
    _build_ic_lora_image_conditionings,
    _compose_ic_lora_video_conditioning,
)


def test_build_ic_lora_image_conditionings_multi_anchor():
    images = _build_ic_lora_image_conditionings("/tmp/char.jpg", 97)
    assert images[0] == ("/tmp/char.jpg", 0, 1.0, IC_LORA_IMAGE_CRF)
    assert images[1] == ("/tmp/char.jpg", 96, 1.0, IC_LORA_IMAGE_CRF)


def test_build_ic_lora_image_conditionings_single_frame():
    images = _build_ic_lora_image_conditionings("/tmp/char.jpg", 1)
    assert len(images) == 1
    assert images[0][1] == 0


def test_compose_ic_lora_video_conditioning_image_only_unchanged(tmp_path: Path):
    motion = tmp_path / "motion.mp4"
    motion.write_bytes(b"not-a-real-mp4")
    vc, cleanup = _compose_ic_lora_video_conditioning(
        [(str(motion), 0.9)],
        identity_image_path=None,
        width=512,
        height=288,
        num_frames=25,
        fps=24.0,
        tmpdir=str(tmp_path),
    )
    assert vc == [(str(motion), 0.9)]
    assert cleanup == []


def test_compose_ic_lora_video_conditioning_adds_identity_hold(tmp_path: Path):
    av = pytest.importorskip("av")
    from ltx_media import encode_image_hold_video

    img = tmp_path / "char.png"
    Image.new("RGB", (64, 48), color=(200, 50, 50)).save(img)
    motion = tmp_path / "motion.mp4"
    motion.write_bytes(b"placeholder")

    vc, cleanup = _compose_ic_lora_video_conditioning(
        [(str(motion), 0.85)],
        identity_image_path=str(img),
        width=64,
        height=48,
        num_frames=25,
        fps=24.0,
        tmpdir=str(tmp_path),
    )
    assert len(vc) == 2
    assert vc[0] == (str(motion), 0.85)
    hold_path = vc[1][0]
    assert vc[1][1] == 1.0
    assert hold_path.endswith("ic_lora_identity_hold.mp4")
    assert cleanup == [hold_path]
    with av.open(hold_path) as container:
        stream = container.streams.video[0]
        frames = sum(1 for _ in container.decode(stream))
    assert frames == 25


def test_encode_image_hold_video_frame_count(tmp_path: Path):
    av = pytest.importorskip("av")
    from ltx_media import encode_image_hold_video

    img = tmp_path / "still.jpg"
    Image.new("RGB", (128, 96), color=(10, 120, 200)).save(img)
    out = tmp_path / "hold.mp4"
    encode_image_hold_video(
        img,
        out,
        width=128,
        height=96,
        num_frames=97,
        fps=24.0,
    )
    with av.open(str(out)) as container:
        stream = container.streams.video[0]
        assert stream.width == 128
        assert stream.height == 96
        frames = sum(1 for _ in container.decode(stream))
    assert frames == 97
