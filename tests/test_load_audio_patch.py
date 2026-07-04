"""Ensure ltx-ws replaces upstream ffmpeg load_audio with PyAV only."""

from __future__ import annotations

import sys
import types

import ltx_media


def test_patch_load_audio_pyav_only(monkeypatch):
    audio_mod = types.SimpleNamespace(
        load_audio=object(),
    )
    fake_pkg = types.SimpleNamespace(utils=types.SimpleNamespace(audio=audio_mod))
    monkeypatch.setitem(sys.modules, "ltx_core_mlx", fake_pkg)
    monkeypatch.setitem(sys.modules, "ltx_core_mlx.utils", fake_pkg.utils)
    monkeypatch.setitem(sys.modules, "ltx_core_mlx.utils.audio", audio_mod)

    stale_pipeline = types.SimpleNamespace(load_audio=audio_mod.load_audio)
    monkeypatch.setitem(sys.modules, "ltx_pipelines_mlx.a2vid_two_stage", stale_pipeline)

    from ltx_mlx_backend import _patch_load_audio_pyav_only

    _patch_load_audio_pyav_only()
    assert audio_mod.load_audio is ltx_media.load_audio_for_inference
    assert stale_pipeline.load_audio is ltx_media.load_audio_for_inference

    # Idempotent
    _patch_load_audio_pyav_only()
    assert audio_mod.load_audio is ltx_media.load_audio_for_inference
