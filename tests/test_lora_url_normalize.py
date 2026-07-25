"""LoRA URL normalization and Path()-mangle recovery."""

from __future__ import annotations

from pathlib import Path

import pytest


CROSSVIEW = (
    "https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-CrossView-Prompt/"
    "resolve/main/LTX2.3-22B_IC-LoRA-CrossView-Prompt_v0.9_13700.safetensors"
)


def test_normalize_lora_spec_passthrough():
    from ltx_mlx_backend import _normalize_lora_spec

    assert _normalize_lora_spec(CROSSVIEW) == CROSSVIEW


def test_normalize_lora_spec_repairs_path_collapsed_url():
    from ltx_mlx_backend import _normalize_lora_spec

    collapsed = CROSSVIEW.replace("https://", "https:/", 1)
    assert collapsed.startswith("https:/") and not collapsed.startswith("https://")
    assert _normalize_lora_spec(collapsed) == CROSSVIEW


def test_normalize_lora_spec_repairs_cwd_prefixed_url(tmp_path: Path, monkeypatch):
    from ltx_mlx_backend import _normalize_lora_spec

    monkeypatch.chdir(tmp_path)
    mangled = str(Path(CROSSVIEW).expanduser().resolve())
    assert mangled.startswith(str(tmp_path))
    assert "https:/" in mangled
    assert _normalize_lora_spec(mangled) == CROSSVIEW


def test_read_custom_loras_repairs_and_persists(tmp_path: Path):
    from web_ui import _read_custom_loras, _write_custom_loras, read_web_settings

    mangled = str(Path(CROSSVIEW).expanduser().resolve())
    _write_custom_loras(
        tmp_path,
        [{"id": "custom_x", "label": "CrossView", "spec": mangled, "scale": 1.2}],
    )
    entries = _read_custom_loras(tmp_path)
    assert entries[0]["spec"] == CROSSVIEW
    saved = read_web_settings(tmp_path)["custom_loras"][0]["spec"]
    assert saved == CROSSVIEW
