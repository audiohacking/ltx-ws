"""Integration tests for PyAV video encode + audio mux (a2v output path)."""

from __future__ import annotations

import struct
import wave
from fractions import Fraction
from pathlib import Path

import pytest

import ltx_media


def _write_wav(path: Path, *, seconds: float, rate: int = 44100) -> None:
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(struct.pack("<h", 0) * int(rate * seconds))


def _write_h264_mp4(
    path: Path,
    *,
    width: int = 128,
    height: int = 96,
    frames: int = 24,
    fps: float = 24.0,
) -> None:
    """Write a silent libx264 MP4 the same way stream_decoder_latent_to_mp4 does."""
    pytest.importorskip("av")
    import av

    fps_frac = ltx_media._pyav_frame_rate(fps)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps_frac, width=width, height=height)
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}
        stream.time_base = Fraction(fps_frac.denominator, fps_frac.numerator)
        for i in range(frames):
            frame = av.VideoFrame(width, height, "yuv420p")
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def test_mux_audio_into_video(tmp_path: Path):
    pytest.importorskip("av")
    import av

    video = tmp_path / "silent.mp4"
    audio = tmp_path / "track.wav"
    output = tmp_path / "a2v.mp4"
    _write_h264_mp4(video, frames=48, fps=24.0)
    _write_wav(audio, seconds=2.0)

    ltx_media.mux_audio_into_video(video, audio, output, duration_s=48 / 24.0)

    assert output.is_file()
    assert output.stat().st_size > 0
    with av.open(str(output)) as container:
        assert container.streams.video
        assert container.streams.audio
        video_stream = container.streams.video[0]
        audio_stream = container.streams.audio[0]
        assert len(list(container.demux(video_stream))) > 0
        assert len(list(container.demux(audio_stream))) > 0
    with av.open(str(output)) as container:
        assert len(list(container.decode(container.streams.audio[0]))) > 0


def test_stream_decoder_mux_path(tmp_path: Path):
    """Exercise video_tmp + mux_audio_into_video, matching a2v decode output."""
    pytest.importorskip("av")
    import av

    video_tmp = tmp_path / "video_only.mp4"
    audio = tmp_path / "audio.wav"
    output = tmp_path / "final.mp4"
    frame_count = 12
    fps = 24.0

    _write_h264_mp4(video_tmp, frames=frame_count, fps=fps)
    _write_wav(audio, seconds=5.0)

    ltx_media.mux_audio_into_video(
        video_tmp,
        audio,
        output,
        duration_s=frame_count / fps,
    )

    with av.open(str(output)) as container:
        video_stream = container.streams.video[0]
        decoded_frames = len(list(container.decode(video_stream)))
        assert decoded_frames == frame_count
        assert container.streams.audio


def test_concat_videos_copy(tmp_path: Path):
    pytest.importorskip("av")
    import av

    a = tmp_path / "a.mp4"
    b = tmp_path / "b.mp4"
    out = tmp_path / "joined.mp4"
    _write_h264_mp4(a, frames=6, fps=24.0)
    _write_h264_mp4(b, frames=6, fps=24.0)

    ltx_media.concat_videos([a, b], out)

    with av.open(str(out)) as container:
        assert len(list(container.decode(container.streams.video[0]))) == 12
