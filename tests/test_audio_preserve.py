"""Reference audio preservation helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


def test_maybe_preserve_reference_audio_skips_silent_source(tmp_path: Path):
    from ltx_mlx_backend import _maybe_preserve_reference_audio

    out = tmp_path / "out.mp4"
    out.write_bytes(b"video")
    ref = tmp_path / "ref.mp4"

    with patch("ltx_media.probe_video_info") as probe:
        probe.return_value = type("Info", (), {"has_audio": False})()
        _maybe_preserve_reference_audio(str(out), [str(ref)], job_id="job-1")
    assert out.read_bytes() == b"video"
