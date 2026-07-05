"""IC-LoRA request wiring for Web UI and backend params."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest


def test_build_params_includes_image_for_ic_lora(tmp_path: Path):
    from web_ui import IC_LORA_DEFAULT_SPEC, _build_params_from_request

    img = tmp_path / "char.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")

    body = {
        "mode": "ic_lora",
        "prompt": "cinematic portrait",
        "image_path": str(img),
        "video_conditioning": [[str(tmp_path / "motion.mp4"), 1.0]],
        "lora_specs": [[IC_LORA_DEFAULT_SPEC, 1.0]],
    }
    (tmp_path / "motion.mp4").write_bytes(b"fake")

    params = _build_params_from_request(body)
    assert params.generation_mode == "ic_lora"
    assert params.initial_image is not None
    assert params.lora_specs == [(IC_LORA_DEFAULT_SPEC, 1.0)]
    assert len(params.video_conditioning_specs) == 1


def test_resolve_ic_lora_video_conditioning_from_upload(tmp_path: Path):
    video = tmp_path / "ref.mp4"
    video.write_bytes(b"fake")

    from web_ui import AppState, _resolve_ic_lora_video_conditioning

    state = MagicMock(spec=AppState)
    state.clips = {}

    body = {
        "mode": "ic_lora",
        "conditioning_video_path": str(video),
        "conditioning_video_scale": 0.85,
    }
    out = _resolve_ic_lora_video_conditioning(state, body)
    assert out["video_conditioning"] == [[str(video), 0.85]]


def test_resolve_ic_lora_video_conditioning_from_clip(tmp_path: Path):
    from web_ui import AppState, ClipRecord, RunStatus, _resolve_ic_lora_video_conditioning

    out_dir = tmp_path / "outputs"
    out_dir.mkdir()
    clip_file = out_dir / "clip0.mp4"
    clip_file.write_bytes(b"fake")

    clip = ClipRecord(
        id="clip-1",
        chain_id="chain-1",
        clip_index=0,
        prompt="test",
        label="clip 1",
        video_url="/api/clips/clip-1/video",
        mode="generate",
        status=RunStatus.DONE.value,
        filename="clip0.mp4",
        created_at="2026-01-01T00:00:00",
    )
    state = MagicMock(spec=AppState)
    state.output_dir = out_dir
    state.clips = {"clip-1": clip}

    body = {"mode": "ic_lora", "conditioning_clip_id": "clip-1"}
    out = _resolve_ic_lora_video_conditioning(state, body)
    assert out["video_conditioning"] == [[str(clip_file.resolve()), 1.0]]


def test_apply_ic_lora_defaults_injects_hdr_lora():
    from web_ui import IC_LORA_DEFAULT_SCALE, IC_LORA_DEFAULT_SPEC, _apply_ic_lora_defaults

    out = _apply_ic_lora_defaults({"mode": "ic_lora", "prompt": "test"})
    assert out["lora_specs"] == [[IC_LORA_DEFAULT_SPEC, IC_LORA_DEFAULT_SCALE]]

    unchanged = _apply_ic_lora_defaults({"mode": "generate", "lora_specs": [["x", 1.0]]})
    assert unchanged["lora_specs"] == [["x", 1.0]]


def test_apply_ic_lora_defaults_union_for_motion_transfer(tmp_path: Path):
    from web_ui import (
        IC_LORA_DEFAULT_SCALE,
        IC_LORA_UNION_MOTION_SPEC,
        _apply_ic_lora_defaults,
    )

    img = tmp_path / "char.jpg"
    img.write_bytes(b"x")
    motion = tmp_path / "motion.mp4"
    motion.write_bytes(b"x")
    out = _apply_ic_lora_defaults(
        {
            "mode": "ic_lora",
            "prompt": "portrait walking",
            "image_path": str(img),
            "video_conditioning": [[str(motion), 1.0]],
        }
    )
    assert out["lora_specs"] == [[IC_LORA_UNION_MOTION_SPEC, IC_LORA_DEFAULT_SCALE]]


def test_apply_ic_lora_defaults_hdr_for_motion_only(tmp_path: Path):
    from web_ui import (
        IC_LORA_DEFAULT_SCALE,
        IC_LORA_DEFAULT_SPEC,
        _apply_ic_lora_defaults,
    )

    motion = tmp_path / "motion.mp4"
    motion.write_bytes(b"x")
    out = _apply_ic_lora_defaults(
        {
            "mode": "ic_lora",
            "prompt": "cinematic scene",
            "video_conditioning": [[str(motion), 1.0]],
        }
    )
    assert out["lora_specs"] == [[IC_LORA_DEFAULT_SPEC, IC_LORA_DEFAULT_SCALE]]


def test_ic_lora_t2v_allows_missing_video_conditioning():
    from web_ui import IC_LORA_DEFAULT_SPEC, _build_params_from_request

    params = _build_params_from_request(
        {
            "mode": "ic_lora",
            "prompt": "sunset over ocean",
            "lora_specs": [[IC_LORA_DEFAULT_SPEC, 1.0]],
        }
    )
    assert params.generation_mode == "ic_lora"
    assert params.video_conditioning_specs == []
