"""Face swap uses IC-LoRA (not LipDub) with a trimmed motion reference."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import pytest


def test_prepare_face_swap_reference_trims_to_requested_frames(tmp_path: Path):
    pytest.importorskip("av")
    from ltx_mlx_backend import _prepare_face_swap_reference_video

    ref = tmp_path / "ref.mp4"
    ref.write_bytes(b"placeholder")

    with patch("ltx_media.probe_video_info") as probe, patch(
        "ltx_media.trim_video_to_spec"
    ) as trim:
        probe.return_value = type(
            "Info",
            (),
            {"num_frames": 889, "fps": 30.0},
        )()
        trim.side_effect = lambda _src, dst, **_: Path(dst).write_bytes(b"trimmed") or Path(
            dst
        )

        out = _prepare_face_swap_reference_video(
            str(ref),
            tmpdir=str(tmp_path),
            num_frames=121,
            width=480,
            height=704,
            fps=24.0,
        )

    trim.assert_called_once()
    assert trim.call_args.kwargs["num_frames"] == 121
    assert trim.call_args.kwargs["width"] == 480
    assert trim.call_args.kwargs["height"] == 704
    assert trim.call_args.kwargs["fps"] == 24.0
    assert out.endswith("face_swap_ref_trimmed.mp4")


def test_head_swap_lora_skips_pose_preprocess():
    from ltx_mlx_backend import _needs_pose_control_preprocessing

    lora = (
        "/loras/Alissonerdx__BFS-Best-Face-Swap-Video/"
        "head_swap_v3_rank_adaptive_fro_098.safetensors",
        0.98,
    )
    with patch("ltx_mlx_backend._ic_lora_reference_downscale_factor", return_value=1):
        with patch("ltx_mlx_backend._ic_lora_uses_hdr_pipeline", return_value=False):
            assert not _needs_pose_control_preprocessing([lora], [("ref.mp4", 1.0)])


def _write_h264_mp4(path: Path, *, frames: int = 25, fps: float = 24.0) -> None:
    pytest.importorskip("av")
    import av

    import ltx_media

    fps_frac = ltx_media._pyav_frame_rate(fps)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=fps_frac, width=128, height=96)
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18"}
        stream.time_base = Fraction(fps_frac.denominator, fps_frac.numerator)
        for i in range(frames):
            frame = av.VideoFrame(128, 96, "yuv420p")
            frame.pts = i
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode(None):
            container.mux(packet)


def test_trim_video_to_spec_limits_frames(tmp_path: Path):
    pytest.importorskip("av")
    import ltx_media

    src = tmp_path / "src.mp4"
    dst = tmp_path / "dst.mp4"
    _write_h264_mp4(src, frames=25, fps=24.0)

    ltx_media.trim_video_to_spec(
        src,
        dst,
        num_frames=9,
        width=64,
        height=48,
        fps=24.0,
    )
    assert dst.is_file() and dst.stat().st_size > 0
    import av

    with av.open(str(dst)) as container:
        stream = container.streams.video[0]
        frames = sum(1 for _ in container.decode(stream))
    assert frames <= 9
    assert stream.width == 64
    assert stream.height == 48
