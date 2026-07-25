"""Ensure IC-LoRA reference video uses PyAV instead of ffprobe/ffmpeg."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import pytest


def _write_h264_mp4(
    path: Path,
    *,
    width: int = 128,
    height: int = 96,
    frames: int = 25,
    fps: float = 24.0,
) -> None:
    pytest.importorskip("av")
    import av

    import ltx_media

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


def test_probe_video_info_pyav(tmp_path: Path):
    pytest.importorskip("ltx_core_mlx")
    import ltx_media

    video = tmp_path / "ref.mp4"
    _write_h264_mp4(video, frames=25, fps=24.0)

    info = ltx_media.probe_video_info(str(video))
    assert info.width == 128
    assert info.height == 96
    assert info.num_frames >= 24
    assert info.fps == pytest.approx(24.0, abs=0.5)


def test_load_video_frames_normalized_pyav(tmp_path: Path):
    pytest.importorskip("ltx_core_mlx")
    import ltx_media

    video = tmp_path / "ref.mp4"
    _write_h264_mp4(video, frames=17, fps=24.0)

    tensor = ltx_media.load_video_frames_normalized(str(video), 48, 64, max_frames=9)
    assert tuple(tensor.shape) == (1, 3, 9, 48, 64)


def test_load_video_frames_normalized_is_float32(tmp_path: Path):
    pytest.importorskip("ltx_core_mlx")
    import mlx.core as mx
    import ltx_media

    video = tmp_path / "ref.mp4"
    _write_h264_mp4(video, frames=5, fps=24.0)
    tensor = ltx_media.load_video_frames_normalized(str(video), 48, 64, max_frames=5)
    assert tensor.dtype == mx.float32


def test_video_io_patch_replaces_ffmpeg_helpers():
    pytest.importorskip("ltx_pipelines_mlx")
    from ltx_core_mlx.utils import ffmpeg as ffmpeg_mod
    from ltx_core_mlx.utils import video as video_mod
    from ltx_mlx_backend import _apply_ltx_mlx_patches
    import ltx_media

    _apply_ltx_mlx_patches(default_fps=24.0)
    assert ffmpeg_mod.probe_video_info is ltx_media.probe_video_info
    assert video_mod.load_video_frames_normalized is ltx_media.load_video_frames_normalized


def test_iclora_utils_uses_pyav_probe():
    pytest.importorskip("ltx_pipelines_mlx")
    from ltx_mlx_backend import _apply_ltx_mlx_patches
    import ltx_media
    from ltx_pipelines_mlx import iclora_utils

    _apply_ltx_mlx_patches(default_fps=24.0)
    assert iclora_utils.probe_video_info is ltx_media.probe_video_info
    assert iclora_utils.load_video_frames_normalized is ltx_media.load_video_frames_normalized


def test_lipdub_uses_pyav_probe():
    pytest.importorskip("ltx_pipelines_mlx")
    from ltx_mlx_backend import _apply_ltx_mlx_patches
    import ltx_media
    from ltx_pipelines_mlx import lipdub

    _apply_ltx_mlx_patches(default_fps=24.0)
    assert lipdub.probe_video_info is ltx_media.probe_video_info
