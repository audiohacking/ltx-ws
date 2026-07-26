"""Retake latent-range helpers and request normalization."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_pixel_frames_to_latent_count():
    from ltx_media import pixel_frames_to_latent_count, vae_compatible_frame_count

    assert vae_compatible_frame_count(97) == 97
    assert vae_compatible_frame_count(100) == 97
    assert pixel_frames_to_latent_count(97) == 13
    assert pixel_frames_to_latent_count(121) == 16


def test_normalize_retake_range_defaults_to_full_rewrite(tmp_path: Path, monkeypatch):
    from web_ui import AppState, _normalize_retake_range

    monkeypatch.setattr("web_ui.media_available", lambda: True)

    def fake_summarize(_path: str):
        return {
            "latent_frames": 13,
            "suggested_start": 1,
            "suggested_end": 13,
        }

    monkeypatch.setattr(
        "ltx_media.summarize_video_for_retake",
        fake_summarize,
        raising=False,
    )
    # Import path used inside the helper
    import ltx_media

    monkeypatch.setattr(ltx_media, "summarize_video_for_retake", fake_summarize)

    video = tmp_path / "src.mp4"
    video.write_bytes(b"fake")
    state = AppState(
        server_url="ws://127.0.0.1:8765/ws",
        output_dir=tmp_path,
        upload_dir=tmp_path,
        preferred_model="auto",
        embedded=True,
    )
    body = {"video_path": str(video), "retake_start": 1, "retake_end": 1}
    _normalize_retake_range(state, body)
    assert body["retake_start"] == 1
    assert body["retake_end"] == 13
    assert body["retake_latent_frames"] == 13


def test_normalize_retake_range_clamps_out_of_bounds(tmp_path: Path, monkeypatch):
    from web_ui import AppState, _normalize_retake_range
    import ltx_media

    monkeypatch.setattr("web_ui.media_available", lambda: True)
    monkeypatch.setattr(
        ltx_media,
        "summarize_video_for_retake",
        lambda _p: {
            "latent_frames": 8,
            "suggested_start": 1,
            "suggested_end": 8,
        },
    )
    video = tmp_path / "src.mp4"
    video.write_bytes(b"fake")
    state = AppState(
        server_url="ws://127.0.0.1:8765/ws",
        output_dir=tmp_path,
        upload_dir=tmp_path,
        preferred_model="auto",
        embedded=True,
    )
    body = {"video_path": str(video), "retake_start": -3, "retake_end": 99}
    _normalize_retake_range(state, body)
    assert body["retake_start"] == 0
    assert body["retake_end"] == 8
