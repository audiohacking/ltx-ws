"""LoRA application on generate pipelines (pending LoRAs must actually fuse)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_apply_pending_loras_sets_attribute_even_when_missing():
    """Regression: hasattr-guard dropped Web UI LoRAs on DistilledPipeline."""
    from ltx_mlx_backend import _apply_pending_loras

    pipe = SimpleNamespace(dit=None, _loaded=False)
    _apply_pending_loras(pipe, [("lora.safetensors", 1.0)])
    assert pipe._pending_loras == [("lora.safetensors", 1.0)]


def test_apply_pending_loras_reloads_when_dit_already_loaded():
    from ltx_mlx_backend import _apply_pending_loras

    loads: list[int] = []

    class Pipe:
        def __init__(self) -> None:
            self.dit = object()  # already loaded without LoRAs
            self._loaded = True
            self._pending_loras = []

        def load(self) -> None:
            loads.append(1)
            self.dit = object()
            self._loaded = True

    pipe = Pipe()
    _apply_pending_loras(pipe, [("a.safetensors", 1.25)])
    assert pipe._pending_loras == [("a.safetensors", 1.25)]
    assert loads == [1]
    assert pipe._loaded is True


def test_apply_pending_loras_skips_ic_lora_owned_paths():
    from ltx_mlx_backend import _apply_pending_loras

    pipe = SimpleNamespace(
        dit=object(),
        _loaded=True,
        _lora_paths=[("crossview.safetensors", 1.25)],
        _pending_loras=[],
    )
    _apply_pending_loras(pipe, [("other.safetensors", 1.0)])
    # IC-LoRA owns fusion; pending must not clobber / reload.
    assert pipe._pending_loras == []


def test_tune_crossview_strength_bumps_low_scale():
    from ltx_mlx_backend import _tune_ic_lora_strengths

    out = _tune_ic_lora_strengths(
        [
            (
                "/loras/Cseti__LTX2.3-22B_IC-LoRA-CrossView-Prompt/x.safetensors",
                1.0,
            )
        ]
    )
    assert out[0][1] == pytest.approx(1.25)


def test_tune_crossview_strength_respects_explicit_high_scale():
    from ltx_mlx_backend import _tune_ic_lora_strengths

    out = _tune_ic_lora_strengths(
        [("CrossView-Prompt_v0.9.safetensors", 1.5)]
    )
    assert out[0][1] == pytest.approx(1.5)


def test_clamp_conditioning_attention_strength():
    from ltx_mlx_backend import _clamp_conditioning_attention_strength

    assert _clamp_conditioning_attention_strength(1.25) == pytest.approx(1.0)
    assert _clamp_conditioning_attention_strength(0.8) == pytest.approx(0.8)
    assert _clamp_conditioning_attention_strength(None) is None
