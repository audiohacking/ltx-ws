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


def test_apply_ic_lora_defaults_keeps_extra_loras():
    from web_ui import (
        IC_LORA_DEFAULT_SCALE,
        IC_LORA_DEFAULT_SPEC,
        IC_LORA_UNION_MOTION_SPEC,
        _apply_ic_lora_defaults,
    )

    extra = "https://example.com/custom-lora.safetensors"
    out = _apply_ic_lora_defaults(
        {
            "mode": "ic_lora",
            "prompt": "test",
            "image_path": "/tmp/char.jpg",
            "video_conditioning": [["/tmp/motion.mp4", 1.0]],
            "lora_specs": [[extra, 0.5], [IC_LORA_UNION_MOTION_SPEC, 1.0]],
        }
    )
    assert out["lora_specs"][0] == [IC_LORA_UNION_MOTION_SPEC, IC_LORA_DEFAULT_SCALE]
    assert [extra, 0.5] in out["lora_specs"]
    assert len(out["lora_specs"]) == 2


def test_apply_ic_lora_defaults_respects_custom_only_crossview():
    """CrossView-style V2V: custom IC-LoRA alone must not get HDR injected."""
    from web_ui import IC_LORA_DEFAULT_SPEC, _apply_ic_lora_defaults

    crossview = (
        "https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-CrossView-Prompt/"
        "resolve/main/LTX2.3-22B_IC-LoRA-CrossView-Prompt_v0.9_13700.safetensors"
    )
    out = _apply_ic_lora_defaults(
        {
            "mode": "ic_lora",
            "prompt": "crossview. new camera angle: to the right, lower, closer.",
            "video_conditioning": [["/tmp/ref.mp4", 1.0]],
            "lora_specs": [[crossview, 1.2]],
        }
    )
    assert out["lora_specs"] == [[crossview, 1.2]]
    assert IC_LORA_DEFAULT_SPEC not in [row[0] for row in out["lora_specs"]]


def test_lora_catalog_includes_custom_entry(tmp_path: Path):
    from web_ui import _lora_catalog, _write_custom_loras

    crossview = (
        "https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-CrossView-Prompt/"
        "resolve/main/LTX2.3-22B_IC-LoRA-CrossView-Prompt_v0.9_13700.safetensors"
    )
    _write_custom_loras(
        tmp_path,
        [
            {
                "id": "custom_crossview",
                "label": "CrossView Prompt",
                "spec": crossview,
                "scale": 1.2,
            }
        ],
    )
    presets, _ = _lora_catalog(tmp_path)
    match = next(p for p in presets if p["id"] == "custom_crossview")
    assert match["spec"] == crossview
    assert match["custom"] is True
    assert match["scale"] == pytest.approx(1.2)
    # Customs are listed after builtins so they remain visible at the end of the menu.
    assert presets.index(match) > presets.index(
        next(p for p in presets if p["id"] == "ic_lora_hdr")
    )


def test_lora_catalog_keeps_custom_when_spec_matches_builtin(tmp_path: Path):
    """Custom entries are keyed by id, so a URL that matches a builtin still appears."""
    from web_ui import IC_LORA_DEFAULT_SPEC, _lora_catalog, _write_custom_loras

    _write_custom_loras(
        tmp_path,
        [
            {
                "id": "custom_hdr_copy",
                "label": "My HDR copy",
                "spec": IC_LORA_DEFAULT_SPEC,
                "scale": 1.0,
            }
        ],
    )
    presets, _ = _lora_catalog(tmp_path)
    ids = [p["id"] for p in presets]
    assert "ic_lora_hdr" in ids
    assert "custom_hdr_copy" in ids


def test_add_custom_lora_reuses_same_spec(tmp_path: Path):
    from web_ui import _lora_catalog, _read_custom_loras, _write_custom_loras

    crossview = (
        "https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-CrossView-Prompt/"
        "resolve/main/LTX2.3-22B_IC-LoRA-CrossView-Prompt_v0.9_13700.safetensors"
    )
    _write_custom_loras(
        tmp_path,
        [{"id": "custom_abc12345", "label": "Old", "spec": crossview, "scale": 1.0}],
    )

    # Exercise catalog + read path used by add_custom_lora dedupe.
    presets, _ = _lora_catalog(tmp_path)
    assert any(p["id"] == "custom_abc12345" for p in presets)
    entries = _read_custom_loras(tmp_path)
    existing = next(e for e in entries if e["spec"] == crossview)
    assert existing["id"] == "custom_abc12345"


def test_build_params_passes_reference_strength(tmp_path: Path):
    from web_ui import IC_LORA_DEFAULT_SPEC, _build_params_from_request

    motion = tmp_path / "ref.mp4"
    motion.write_bytes(b"x")
    params = _build_params_from_request(
        {
            "mode": "ic_lora",
            "prompt": "crossview. new camera angle: to the left, higher, further.",
            "video_conditioning": [[str(motion), 0.9]],
            "lora_specs": [[IC_LORA_DEFAULT_SPEC, 1.0]],
            "reference_strength": 1.25,
        }
    )
    assert params.reference_strength == pytest.approx(1.25)


def test_apply_ic_lora_defaults_strips_alternate_builtin():
    from web_ui import (
        IC_LORA_DEFAULT_SCALE,
        IC_LORA_DEFAULT_SPEC,
        IC_LORA_UNION_MOTION_SPEC,
        _apply_ic_lora_defaults,
    )

    out = _apply_ic_lora_defaults(
        {
            "mode": "ic_lora",
            "prompt": "v2v",
            "video_conditioning": [["/tmp/motion.mp4", 1.0]],
            "lora_specs": [
                [IC_LORA_UNION_MOTION_SPEC, 1.0],
                [IC_LORA_DEFAULT_SPEC, 1.0],
            ],
        }
    )
    specs = [row[0] for row in out["lora_specs"]]
    assert specs == [IC_LORA_DEFAULT_SPEC]


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
