"""Reuse already-downloaded LoRA weights instead of re-downloading."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

CROSSVIEW = (
    "https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-CrossView-Prompt/"
    "resolve/main/LTX2.3-22B_IC-LoRA-CrossView-Prompt_v0.9_13700.safetensors"
)
CROSSVIEW_NAME = "LTX2.3-22B_IC-LoRA-CrossView-Prompt_v0.9_13700.safetensors"


def test_lora_cached_path_finds_file_under_alternate_cache_dir(
    tmp_path: Path, monkeypatch
):
    from ltx_mlx_backend import _lora_cached_path

    monkeypatch.setattr("ltx_mlx_backend._local_lora_cache_dir", lambda: tmp_path)
    # Prior download landed under a different folder name (e.g. manual / old layout).
    orphan = tmp_path / "misc_downloads" / CROSSVIEW_NAME
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"lora-bytes")

    hit = _lora_cached_path(CROSSVIEW)
    assert hit is not None
    assert hit.is_file()
    assert hit.read_bytes() == b"lora-bytes"
    # Promoted to the canonical HF resolve layout.
    assert "Cseti__LTX2.3-22B_IC-LoRA-CrossView-Prompt" in str(hit)


def test_lora_cached_path_finds_nested_local_dir_layout(tmp_path: Path, monkeypatch):
    from ltx_mlx_backend import _hf_lora_cache_file, _lora_cached_path, _parse_hf_lora_resolve_url

    monkeypatch.setattr("ltx_mlx_backend._local_lora_cache_dir", lambda: tmp_path)
    parsed = _parse_hf_lora_resolve_url(CROSSVIEW)
    assert parsed is not None
    nested = (
        tmp_path
        / parsed.cache_dir_name
        / "snapshots"
        / "abc"
        / CROSSVIEW_NAME
    )
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"nested")

    hit = _lora_cached_path(CROSSVIEW)
    assert hit == _hf_lora_cache_file(parsed)
    assert hit.read_bytes() == b"nested"


def test_ensure_reports_cached_without_download(tmp_path: Path, monkeypatch):
    from web_ui import _ensure_lora_downloaded

    monkeypatch.setattr("ltx_mlx_backend._local_lora_cache_dir", lambda: tmp_path)
    orphan = tmp_path / "old" / CROSSVIEW_NAME
    orphan.parent.mkdir(parents=True)
    orphan.write_bytes(b"already-here")

    with patch("ltx_mlx_backend.hf_hub_download", create=True) as mock_dl:
        with patch("ltx_mlx_backend.urlopen") as mock_url:
            result = _ensure_lora_downloaded(CROSSVIEW)

    assert result["ok"] is True
    assert result["cached"] is True
    assert Path(result["path"]).is_file()
    mock_url.assert_not_called()
    # hf_hub_download may be imported lazily; either way no network download path.
    assert mock_dl.call_count == 0 or all(
        c.kwargs.get("local_files_only") for c in mock_dl.call_args_list
    )


def test_download_skips_when_canonical_exists(tmp_path: Path, monkeypatch):
    from ltx_mlx_backend import (
        _download_hf_lora_resolve,
        _hf_lora_cache_file,
        _parse_hf_lora_resolve_url,
    )

    monkeypatch.setattr("ltx_mlx_backend._local_lora_cache_dir", lambda: tmp_path)
    parsed = _parse_hf_lora_resolve_url(CROSSVIEW)
    assert parsed is not None
    dest = _hf_lora_cache_file(parsed)
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"canonical")

    with patch("ltx_mlx_backend.urlopen") as mock_url:
        with patch("huggingface_hub.hf_hub_download") as mock_dl:
            out = _download_hf_lora_resolve(parsed)

    assert out == dest
    mock_url.assert_not_called()
    mock_dl.assert_not_called()


def test_empty_file_not_treated_as_cache(tmp_path: Path, monkeypatch):
    from ltx_mlx_backend import _lora_cached_path

    monkeypatch.setattr("ltx_mlx_backend._local_lora_cache_dir", lambda: tmp_path)
    empty = tmp_path / "Cseti__LTX2.3-22B_IC-LoRA-CrossView-Prompt" / CROSSVIEW_NAME
    empty.parent.mkdir(parents=True)
    empty.write_bytes(b"")

    assert _lora_cached_path(CROSSVIEW) is None
