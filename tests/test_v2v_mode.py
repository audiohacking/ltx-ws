"""V2V + LoRA mode is separate from IC-LoRA (HDR/Union) defaults."""

from __future__ import annotations

from pathlib import Path

import pytest

CROSSVIEW = (
    "https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-CrossView-Prompt/"
    "resolve/main/LTX2.3-22B_IC-LoRA-CrossView-Prompt_v0.9_13700.safetensors"
)


def test_generation_modes_include_v2v():
    from web_ui import GENERATION_MODES

    ids = [m["id"] for m in GENERATION_MODES]
    assert "v2v" in ids
    assert "ic_lora" in ids


def test_api_mode_maps_v2v_to_ic_lora_pipeline():
    from web_ui import _api_mode

    assert _api_mode("v2v") == "ic_lora"
    assert _api_mode("ic_lora") == "ic_lora"


def test_apply_ic_lora_defaults_skips_v2v_mode():
    from web_ui import _apply_ic_lora_defaults

    body = {
        "mode": "v2v",
        "prompt": "crossview. new camera angle: to the right, lower, closer.",
        "lora_specs": [[CROSSVIEW, 1.25]],
        "conditioning_video_path": "/tmp/ref.mp4",
    }
    out = _apply_ic_lora_defaults(body)
    assert out["lora_specs"] == [[CROSSVIEW, 1.25]]


def test_resolve_v2v_video_conditioning(tmp_path: Path):
    from web_ui import AppState, _resolve_ic_lora_video_conditioning

    ref = tmp_path / "ref.mp4"
    ref.write_bytes(b"fake")
    state = AppState(
        server_url="ws://127.0.0.1:8765/ws",
        output_dir=tmp_path,
        upload_dir=tmp_path,
        preferred_model="auto",
        embedded=True,
    )
    out = _resolve_ic_lora_video_conditioning(
        state,
        {
            "mode": "v2v",
            "conditioning_video_path": str(ref),
            "conditioning_video_scale": 1.0,
        },
    )
    assert out["video_conditioning"][0][0] == str(ref.resolve())
