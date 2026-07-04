"""Integration tests for a2v audio decode (PyAV patch + payload round-trip)."""

from __future__ import annotations

import base64
import struct
import sys
import tempfile
import wave
from pathlib import Path

import pytest

import ltx_media


def _write_stereo_wav(path: Path, *, seconds: float, rate: int = 44100) -> None:
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<hh", 700, -700) * int(rate * seconds))


def _payload_from_file(path: Path) -> dict:
    mime = "audio/wav" if path.suffix == ".wav" else "audio/mpeg"
    raw = path.read_bytes()
    data = base64.b64encode(raw).decode()
    return {
        "name": path.name,
        "mime_type": mime,
        "data_url": f"data:{mime};base64,{data}",
    }


def _write_mono_wav(path: Path, *, seconds: float, rate: int = 44100) -> None:
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<h", 0) * int(rate * seconds))


@pytest.fixture
def wav_174s(tmp_path: Path) -> Path:
    path = tmp_path / "source.wav"
    _write_mono_wav(path, seconds=174.0)
    return path


def test_patch_fixes_stale_pipeline_load_audio_binding(monkeypatch):
    """Pipeline modules bind load_audio at import — patch must update them too."""
    ltx_pipelines = pytest.importorskip("ltx_pipelines_mlx")

    mods = [k for k in list(sys.modules) if k.startswith("ltx_")]
    for name in mods:
        monkeypatch.delitem(sys.modules, name, raising=False)

    import ltx_pipelines_mlx.a2vid_two_stage as a2v  # noqa: WPS433

    assert a2v.load_audio.__module__ == "ltx_core_mlx.utils.audio"

    from ltx_mlx_backend import _patch_load_audio_pyav_only

    _patch_load_audio_pyav_only()

    import ltx_core_mlx.utils.audio as audio_mod

    assert audio_mod.load_audio.__name__ == "load_audio_for_inference"
    assert a2v.load_audio is audio_mod.load_audio
    del ltx_pipelines  # silence unused in skip path


def test_decode_media_roundtrip_loads_trimmed_wav():
    pytest.importorskip("mlx.core")
    pytest.importorskip("ltx_core_mlx")

    from ltx_mlx_backend import _decode_media_input, _patch_load_audio_pyav_only

    _patch_load_audio_pyav_only()

    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "source.wav"
        trimmed = Path(td) / "segment.wav"
        _write_stereo_wav(src, seconds=30.0)
        ltx_media.trim_audio_start(src, trimmed, start_seconds=10.0)

        payload = _payload_from_file(trimmed)
        path, cleanup = _decode_media_input(
            payload,
            temp_prefix="fvserver_audio_",
            default_suffix=".wav",
        )
        assert path is not None
        try:
            import ltx_core_mlx.utils.audio as audio_mod

            loaded = audio_mod.load_audio(
                path,
                target_sample_rate=16000,
                max_duration=2.0,
            )
            assert loaded is not None
            assert int(loaded.waveform.shape[-1]) > 0
        finally:
            if cleanup:
                Path(cleanup).unlink(missing_ok=True)


def test_load_audio_for_inference_with_start_offset(wav_174s: Path):
    pytest.importorskip("mlx.core")
    pytest.importorskip("ltx_core_mlx")

    from ltx_mlx_backend import _patch_load_audio_pyav_only

    _patch_load_audio_pyav_only()

    import ltx_core_mlx.utils.audio as audio_mod

    loaded = audio_mod.load_audio(
        wav_174s,
        target_sample_rate=16000,
        start_time=98.0,
        max_duration=15.0,
    )
    assert loaded is not None
    assert int(loaded.waveform.shape[-1]) == pytest.approx(15 * 16000, rel=0.05)
