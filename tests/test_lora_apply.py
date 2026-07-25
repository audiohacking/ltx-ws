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


def test_apply_pending_loras_skips_face_swap_owned_paths():
    """Face swap owns fusion via ``_head_swap_lora`` / ``_lora_paths`` (CrossView lesson)."""
    from ltx_mlx_backend import _apply_pending_loras

    pipe = SimpleNamespace(
        dit=object(),
        _loaded=True,
        _lora_paths=[("head_swap_v3.safetensors", 0.98)],
        _head_swap_lora=[("head_swap_v3.safetensors", 0.98)],
        _pending_loras=[],
    )
    loads: list[int] = []

    def load() -> None:
        loads.append(1)

    pipe.load = load
    _apply_pending_loras(pipe, None)
    assert pipe._pending_loras == []
    assert loads == []


def test_build_face_swap_lora_stack_prepends_distilled_dynamic(monkeypatch, tmp_path):
    from ltx_mlx_backend import (
        FACE_SWAP_DISTILLED_DYNAMIC_SCALE,
        _build_face_swap_lora_stack,
    )

    distilled = tmp_path / "ltx-2.3-22b-distilled-lora-dynamic_fro09_avg_rank_105_bf16.safetensors"
    distilled.write_bytes(b"lora")
    head = tmp_path / "head_swap_v3_rank_adaptive_fro_098.safetensors"
    head.write_bytes(b"head")

    monkeypatch.delenv("LTX_WS_FACE_SWAP_NO_DISTILLED_LORA", raising=False)
    monkeypatch.delenv("LTX_WS_FACE_SWAP_DISTILLED_LORA", raising=False)

    stack = _build_face_swap_lora_stack(
        [(str(head), 0.98)],
        model_dir=tmp_path,
    )
    assert len(stack) == 2
    assert stack[0] == (str(distilled.resolve()), FACE_SWAP_DISTILLED_DYNAMIC_SCALE)
    assert stack[1][0] == str(head)
    assert stack[1][1] == pytest.approx(0.98)


def test_build_face_swap_lora_stack_can_skip_distilled(monkeypatch, tmp_path):
    from ltx_mlx_backend import _build_face_swap_lora_stack

    head = tmp_path / "head_swap_v3.safetensors"
    head.write_bytes(b"head")
    monkeypatch.setenv("LTX_WS_FACE_SWAP_NO_DISTILLED_LORA", "1")

    stack = _build_face_swap_lora_stack([(str(head), 0.98)], model_dir=tmp_path)
    assert stack == [(str(head), 0.98)]


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


def test_should_use_control_aware_refine_for_crossview():
    from ltx_mlx_backend import _should_use_control_aware_refine

    assert (
        _should_use_control_aware_refine(
            [("/loras/Cseti__CrossView-Prompt/x.safetensors", 1.25)]
        )
        is True
    )


def test_should_use_control_aware_refine_for_empty_loras():
    from ltx_mlx_backend import _should_use_control_aware_refine

    assert _should_use_control_aware_refine([]) is True


def test_should_use_control_aware_refine_skips_hdr():
    from ltx_mlx_backend import _should_use_control_aware_refine

    assert (
        _should_use_control_aware_refine(
            [("/weights/LTX-2.3-22b-IC-LoRA-HDR.safetensors", 1.0)]
        )
        is False
    )


def test_iclora_supports_control_aware_refine_on_current_install():
    """Pinned ltx-2-mlx >= 0.14.17 exposes upsample_only / refine_steps."""
    from ltx_mlx_backend import _iclora_supports_control_aware_refine

    assert _iclora_supports_control_aware_refine() is True
