"""Tests for PyAV-backed ltx_media helpers."""

from __future__ import annotations

import struct
import tempfile
import wave
from pathlib import Path

import pytest

import ltx_media


def _write_wav(path: Path, *, seconds: float, rate: int = 44100) -> None:
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<h", 0) * int(rate * seconds))


@pytest.fixture
def wav_174s(tmp_path: Path) -> Path:
    path = tmp_path / "source.wav"
    _write_wav(path, seconds=174.0)
    return path


def test_media_available():
    assert ltx_media.media_available() is True


def test_probe_audio_duration(wav_174s: Path):
    duration = ltx_media.probe_audio_duration(wav_174s)
    assert duration is not None
    assert duration == pytest.approx(174.0, abs=0.2)


def test_trim_audio_start_leaves_remainder(wav_174s: Path, tmp_path: Path):
    out = tmp_path / "trimmed.wav"
    ltx_media.trim_audio_start(wav_174s, out, start_seconds=98.0)
    trimmed = ltx_media.probe_audio_duration(out)
    assert trimmed is not None
    assert trimmed == pytest.approx(76.0, abs=0.5)
    assert out.stat().st_size > ltx_media._MIN_WAV_BYTES


def test_trim_audio_to_temp():
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "in.wav"
        _write_wav(src, seconds=10.0)
        out, temp_dir = ltx_media.trim_audio_to_temp(str(src), 3.0)
        assert out.parent == temp_dir
        duration = ltx_media.probe_audio_duration(out)
        assert duration == pytest.approx(7.0, abs=0.5)


def test_split_audio_segments(wav_174s: Path, tmp_path: Path):
    out_dir = tmp_path / "segments"
    segments = ltx_media.split_audio(
        wav_174s,
        out_dir,
        segment_seconds=15.0,
        required_segments=3,
    )
    assert len(segments) >= 3
    for seg in segments[:3]:
        assert seg.is_file()
        assert ltx_media.probe_audio_duration(seg) is not None


def test_split_audio_insufficient_length(tmp_path: Path):
    src = tmp_path / "short.wav"
    _write_wav(src, seconds=5.0)
    out_dir = tmp_path / "segments"
    with pytest.raises(RuntimeError, match="produced 1 segment"):
        ltx_media.split_audio(
            src,
            out_dir,
            segment_seconds=15.0,
            required_segments=3,
        )


def test_trim_near_end_produces_short_remainder(wav_174s: Path, tmp_path: Path):
    out = tmp_path / "tail.wav"
    ltx_media.trim_audio_start(wav_174s, out, start_seconds=170.0)
    trimmed = ltx_media.probe_audio_duration(out)
    assert trimmed is not None
    assert trimmed < 15.0
