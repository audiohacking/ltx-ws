"""Hugging Face bucket LoRA resolve URLs (public IC-LoRA HDR)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from web_ui import IC_LORA_DEFAULT_SPEC


def test_ic_lora_default_spec_uses_public_bucket():
    assert "buckets/audiohacking/LTX-2.3-22b-IC-LoRA-HDR-bucket" in IC_LORA_DEFAULT_SPEC
    assert IC_LORA_DEFAULT_SPEC.endswith("ltx-2.3-22b-ic-lora-hdr-0.9.safetensors")


def test_parse_hf_bucket_resolve_url():
    from ltx_mlx_backend import _hf_lora_cache_file, _parse_hf_lora_resolve_url

    parsed = _parse_hf_lora_resolve_url(IC_LORA_DEFAULT_SPEC)
    assert parsed is not None
    assert parsed.cache_dir_name == "audiohacking__LTX-2.3-22b-IC-LoRA-HDR-bucket"
    assert parsed.filename == "ltx-2.3-22b-ic-lora-hdr-0.9.safetensors"
    assert parsed.repo_id is None
    cache_path = _hf_lora_cache_file(parsed)
    assert cache_path.name == "ltx-2.3-22b-ic-lora-hdr-0.9.safetensors"


def test_apply_ic_lora_defaults_uses_public_bucket():
    from web_ui import IC_LORA_DEFAULT_SCALE, _apply_ic_lora_defaults

    out = _apply_ic_lora_defaults({"mode": "ic_lora", "prompt": "hdr"})
    assert out["lora_specs"] == [[IC_LORA_DEFAULT_SPEC, IC_LORA_DEFAULT_SCALE]]


def test_download_hf_bucket_lora_to_cache(tmp_path: Path, monkeypatch):
    from ltx_mlx_backend import _download_hf_lora_resolve, _parse_hf_lora_resolve_url

    parsed = _parse_hf_lora_resolve_url(IC_LORA_DEFAULT_SPEC)
    assert parsed is not None

    monkeypatch.setattr(
        "ltx_mlx_backend._local_lora_cache_dir",
        lambda: tmp_path,
    )

    payload = b"\x00" * 128
    response = MagicMock()
    response.read.side_effect = [payload, b""]
    response.__enter__.return_value = response
    response.__exit__.return_value = False

    with patch("ltx_mlx_backend.urlopen", return_value=response):
        dest = _download_hf_lora_resolve(parsed)

    assert dest.is_file()
    assert dest.read_bytes() == payload
    again = _download_hf_lora_resolve(parsed)
    assert again == dest
