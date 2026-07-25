"""LipDub mode request wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

LIPDUB_TEST_SPEC = (
    "https://huggingface.co/buckets/audiohacking/LTX-2.3-22b-IC-LoRA-LipDub-bucket/"
    "resolve/ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors"
)


def test_lora_catalog_includes_lipdub_public_bucket():
    from web_ui import LIPDUB_DEFAULT_SPEC, LIPDUB_PRESET_ID, _lora_catalog

    presets, _ = _lora_catalog(None)
    match = next(p for p in presets if p["id"] == LIPDUB_PRESET_ID)
    assert "buckets/audiohacking/LTX-2.3-22b-IC-LoRA-LipDub-bucket" in match["spec"]
    assert match["spec"] == LIPDUB_DEFAULT_SPEC
    assert match["scale"] == pytest.approx(1.0)


def test_lora_catalog_lipdub_env_override(monkeypatch):
    from web_ui import LIPDUB_PRESET_ID, _lora_catalog

    monkeypatch.setenv("LTX_WS_LIPDUB_LORA", "/models/lipdub.safetensors")
    presets, _ = _lora_catalog(None)
    match = next(p for p in presets if p["id"] == LIPDUB_PRESET_ID)
    assert match["spec"] == "/models/lipdub.safetensors"


def test_builtin_lipdub_spec_uses_public_bucket(monkeypatch):
    from web_ui import LIPDUB_DEFAULT_SPEC, _builtin_lipdub_spec

    monkeypatch.delenv("LTX_WS_LIPDUB_LORA", raising=False)
    assert _builtin_lipdub_spec() == LIPDUB_DEFAULT_SPEC


def test_build_params_lipdub_video_only(tmp_path: Path):
    from web_ui import _build_params_from_request

    video = tmp_path / "ref.mp4"
    video.write_bytes(b"mp4")

    params = _build_params_from_request(
        {
            "mode": "lipdub",
            "prompt": "Hello, this is the new dialogue.",
            "video_path": str(video),
            "lora_specs": [[LIPDUB_TEST_SPEC, 1.0]],
        }
    )
    assert params.generation_mode == "lipdub"
    assert params.source_video == str(video.resolve())
    assert params.audio_input is None
    assert len(params.lora_specs) == 1


def test_build_params_lipdub_with_voice_tone_audio(tmp_path: Path):
    from web_ui import _build_params_from_request

    video = tmp_path / "ref.mp4"
    video.write_bytes(b"mp4")
    audio = tmp_path / "voice.wav"
    audio.write_bytes(b"wav")

    params = _build_params_from_request(
        {
            "mode": "lipdub",
            "prompt": "New dialogue line",
            "video_path": str(video),
            "audio_path": str(audio),
            "lora_specs": [[LIPDUB_TEST_SPEC, 1.0]],
        }
    )
    assert params.audio_input is not None


def test_build_params_lipdub_optional_anchor_image(tmp_path: Path):
    from web_ui import _build_params_from_request

    video = tmp_path / "ref.mp4"
    video.write_bytes(b"mp4")
    anchor = tmp_path / "anchor.jpg"
    anchor.write_bytes(b"jpeg")

    params = _build_params_from_request(
        {
            "mode": "lipdub",
            "prompt": "New line of dialogue",
            "video_path": str(video),
            "image_path": str(anchor),
            "lora_specs": [[LIPDUB_TEST_SPEC, 1.0]],
        }
    )
    assert params.initial_image == str(anchor.resolve())
    assert params.audio_input is None


def test_lipdub_muxes_separate_voice_audio_when_provided(tmp_path: Path, monkeypatch):
    from ltx_mlx_backend import _prepare_lipdub_reference_video

    video = tmp_path / "ref.mp4"
    audio = tmp_path / "voice.wav"
    video.write_text("video")
    audio.write_text("audio")
    muxed: list[tuple] = []

    def fake_mux(v, a, out, *, duration_s):
        muxed.append((v, a, out, duration_s))
        Path(out).write_text("muxed")
        return Path(out)

    class FakeInfo:
        duration = 2.0
        num_frames = 48
        fps = 24.0
        has_audio = False

    monkeypatch.setattr("ltx_media.media_available", lambda: True)
    monkeypatch.setattr("ltx_media.probe_video_info", lambda _p: FakeInfo())
    monkeypatch.setattr("ltx_media.mux_audio_into_video", fake_mux)

    out, cleanup = _prepare_lipdub_reference_video(
        str(video),
        str(audio),
        tmpdir=str(tmp_path),
    )
    assert muxed
    assert out.endswith("lipdub_ref_with_voice_audio.mp4")
    assert cleanup == [out]


def test_lipdub_requires_audio_when_video_has_none(tmp_path: Path, monkeypatch):
    from ltx_mlx_backend import _prepare_lipdub_reference_video

    class FakeInfo:
        duration = 2.0
        num_frames = 48
        fps = 24.0
        has_audio = False

    monkeypatch.setattr("ltx_media.media_available", lambda: True)
    monkeypatch.setattr("ltx_media.probe_video_info", lambda _p: FakeInfo())

    with pytest.raises(RuntimeError, match="voice-tone audio"):
        _prepare_lipdub_reference_video(
            str(tmp_path / "silent.mp4"),
            None,
            tmpdir=str(tmp_path),
        )


def test_format_lora_download_error_gated_lipdub():
    from ltx_mlx_backend import format_lora_download_error

    msg = format_lora_download_error(
        RuntimeError("403 Client Error: Cannot access gated repo"),
        "https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-LipDub/resolve/main/x.safetensors",
    )
    assert "gated" in msg.lower()
    assert "HF_TOKEN" in msg
