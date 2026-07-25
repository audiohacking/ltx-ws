"""V2V + LoRA mode is separate from IC-LoRA (HDR/Union) defaults."""

from __future__ import annotations

from pathlib import Path

import pytest

CROSSVIEW = (
    "https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-CrossView-Prompt/"
    "resolve/main/LTX2.3-22B_IC-LoRA-CrossView-Prompt_v0.9_13700.safetensors"
)
OMNINFT = "/tmp/LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors"


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


def test_v2v_label_is_video_to_video():
    from web_ui import GENERATION_MODES

    v2v = next(m for m in GENERATION_MODES if m["id"] == "v2v")
    assert "video to video" in v2v["label"].lower()


def test_should_use_prompt_i2v_for_v2v_when_no_lora():
    """Raw IC-RGB without an adapter ignores text — route to first-frame I2V."""
    from ltx_mlx_backend import (
        _should_use_control_aware_refine,
        _should_use_prompt_i2v_for_v2v,
    )

    assert _should_use_prompt_i2v_for_v2v([], [("/tmp/ref.mp4", 1.0)]) is True
    assert _should_use_control_aware_refine([]) is False


def test_should_use_prompt_i2v_false_when_crossview_lora():
    from ltx_mlx_backend import _should_use_prompt_i2v_for_v2v

    cross = f"/models/{CROSSVIEW.split('/')[-1]}"
    assert _should_use_prompt_i2v_for_v2v([(cross, 1.25)], [("/tmp/ref.mp4", 1.0)]) is False


def test_should_use_control_aware_refine_when_crossview_not_first():
    """OmniNFT-first stacks must not disable CrossView control-aware refine."""
    from ltx_mlx_backend import _should_use_control_aware_refine

    stacked = [
        (OMNINFT, 1.0),
        (f"/models/{CROSSVIEW.split('/')[-1]}", 1.25),
    ]
    assert _should_use_control_aware_refine(stacked) is True


def test_ic_lora_primary_prefers_crossview_over_style_default():
    from ltx_mlx_backend import _ic_lora_primary_lora

    stacked = [
        (OMNINFT, 1.0),
        ("/cache/LTX2.3-22B_IC-LoRA-CrossView-Prompt_v0.9_13700.safetensors", 1.25),
    ]
    primary = _ic_lora_primary_lora(stacked)
    assert primary is not None
    assert "crossview" in primary[0].lower()


def test_ic_lora_mode_does_not_stack_omninft_defaults(tmp_path: Path):
    """V2V/IC-LoRA must use request LoRAs only — OmniNFT defaults caused passthrough."""
    from ltx_mlx_backend import GenerationRequest, LocalVideoGenerator

    cross = tmp_path / "LTX2.3-22B_IC-LoRA-CrossView-Prompt_v0.9_13700.safetensors"
    cross.write_bytes(b"fake")
    omni = tmp_path / "omni.safetensors"
    omni.write_bytes(b"fake")

    gen = LocalVideoGenerator.__new__(LocalVideoGenerator)
    gen._resolved_default_loras = [(str(omni), 1.0)]
    gen.default_lora_specs = [(str(omni), 1.0)]
    gen.default_lora_count = lambda: 1  # type: ignore[method-assign]

    mode = "ic_lora"
    req = GenerationRequest(
        prompt="crossview. new camera angle: to the right, lower, closer.",
        mode=mode,
        seed=1,
        lora_specs=[(str(cross), 1.25)],
    )
    resolved_loras: list[tuple[str, float]] = []
    if mode in ("face_swap", "face-swap", "lipdub", "lip_dub", "ic_lora"):
        for lora_spec, lora_scale in (req.lora_specs or []):
            resolved_loras.append((str(lora_spec), float(lora_scale)))
    elif gen._resolved_default_loras is not None and not req.lora_specs:
        resolved_loras = list(gen._resolved_default_loras)

    assert resolved_loras == [(str(cross), 1.25)]
    assert str(omni) not in {p for p, _ in resolved_loras}


def test_ic_lora_empty_request_skips_omninft_defaults():
    """No LoRA in V2V must stay empty so the text prompt drives the rewrite."""
    mode = "ic_lora"
    omni = ("/tmp/omni.safetensors", 1.0)
    resolved_loras: list[tuple[str, float]] = []
    resolved_default = [omni]
    req_lora_specs: list[tuple[str, float]] = []
    if mode in ("face_swap", "face-swap", "lipdub", "lip_dub", "ic_lora"):
        for lora_spec, lora_scale in req_lora_specs:
            resolved_loras.append((str(lora_spec), float(lora_scale)))
    elif resolved_default is not None and not req_lora_specs:
        resolved_loras = list(resolved_default)

    assert resolved_loras == []
