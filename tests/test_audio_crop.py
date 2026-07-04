"""Tests for audio crop + scratch paths."""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import pytest

import ltx_media
from ltx_paths import configure_scratch_root, scratch_root


def _write_stereo_wav(path: Path, *, seconds: float, rate: int = 44100) -> None:
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<hh", 500, -500) * int(rate * seconds))


def test_apply_audio_start_crops_and_clears_offset(tmp_path: Path, monkeypatch):
    configure_scratch_root(tmp_path / "scratch")
    src = tmp_path / "song.wav"
    _write_stereo_wav(src, seconds=30.0)

    from web_ui import _apply_audio_start_offset

    body = {
        "mode": "a2v",
        "audio_path": str(src),
        "audio_start_seconds": 10.0,
        "duration_seconds": 5.0,
        "audiocontinue": False,
    }
    new_body, temps = _apply_audio_start_offset(body)
    assert new_body["audio_start_seconds"] == 0
    cropped = Path(new_body["audio_path"])
    assert cropped.is_file()
    assert str(scratch_root()) in str(cropped)
    duration = ltx_media.probe_audio_duration(cropped)
    assert duration is not None
    assert duration == pytest.approx(20.0, abs=0.5)
    assert temps


def test_local_file_ref_skips_base64(tmp_path: Path):
    from web_ui import _local_file_ref

    src = tmp_path / "a.mp3"
    src.write_bytes(b"not-real-mp3")
    ref = _local_file_ref(str(src))
    assert ref == str(src.resolve())


def test_trim_uses_scratch_root(tmp_path: Path):
    configure_scratch_root(tmp_path / "scratch")
    src = tmp_path / "in.wav"
    _write_stereo_wav(src, seconds=12.0)
    out, temp_dir = ltx_media.trim_audio_to_temp(str(src), 4.0)
    assert str(scratch_root()) in str(out)
    assert str(scratch_root()) in str(temp_dir)
    assert ltx_media.probe_audio_duration(out) == pytest.approx(8.0, abs=0.5)


def test_crop_then_load_for_inference_stereo(tmp_path: Path):
    pytest.importorskip("mlx.core")
    pytest.importorskip("ltx_core_mlx")

    configure_scratch_root(tmp_path / "scratch")
    src = tmp_path / "song.wav"
    _write_stereo_wav(src, seconds=30.0)
    cropped, _temp = ltx_media.trim_audio_to_temp(str(src), 10.0)

    from ltx_mlx_backend import _patch_load_audio_pyav_only

    _patch_load_audio_pyav_only()
    import ltx_core_mlx.utils.audio as audio_mod

    loaded = audio_mod.load_audio(str(cropped), target_sample_rate=16000, max_duration=2.0)
    assert loaded is not None
    assert int(loaded.waveform.shape[-1]) > 0
