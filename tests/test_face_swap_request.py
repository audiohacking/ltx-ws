"""Face swap mode request wiring."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_lora_catalog_includes_face_swap_preset():
    from web_ui import FACE_SWAP_DEFAULT_SPEC, FACE_SWAP_PRESET_ID, _lora_catalog

    presets, _ = _lora_catalog(None)
    match = next(p for p in presets if p["id"] == FACE_SWAP_PRESET_ID)
    assert match["spec"] == FACE_SWAP_DEFAULT_SPEC
    assert match["scale"] == pytest.approx(0.98)


def test_build_params_face_swap(tmp_path: Path):
    from web_ui import _build_params_from_request

    face = tmp_path / "face.jpg"
    face.write_bytes(b"jpeg")
    video = tmp_path / "ref.mp4"
    video.write_bytes(b"mp4")

    params = _build_params_from_request(
        {
            "mode": "face_swap",
            "prompt": "person speaking to camera",
            "image_path": str(face),
            "video_path": str(video),
            "lora_specs": [[
                "https://huggingface.co/Alissonerdx/BFS-Best-Face-Swap-Video/"
                "resolve/main/ltx-2.3/head_swap_v3_rank_adaptive_fro_098.safetensors",
                0.98,
            ]],
        }
    )
    assert params.generation_mode == "face_swap"
    assert params.initial_image == str(face.resolve())
    assert params.source_video == str(video.resolve())
    assert len(params.lora_specs) == 1
