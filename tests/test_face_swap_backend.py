"""Face swap uses BFS V3 composite guide video (LTXVAddGuide path, no IC-LoRA normalize)."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import pytest


def test_format_head_swap_prompt_wraps_plain_text():
    from ltx_face_swap_compose import format_head_swap_prompt

    out = format_head_swap_prompt("person talking to camera")
    assert out.startswith("head_swap:")
    assert "FACE:" in out
    assert "ACTION:" in out
    assert "person talking to camera" in out
    assert "Blue eyes" not in out
    assert "side-panel" not in out.lower()


def test_format_head_swap_prompt_preserves_existing_trigger():
    from ltx_face_swap_compose import format_head_swap_prompt

    original = "head_swap:\n\nFACE:\nBlue eyes.\n\nACTION:\nWaves."
    assert format_head_swap_prompt(original) == original


def test_prepare_face_swap_guide_trims_and_composes(tmp_path: Path):
    from ltx_mlx_backend import _prepare_face_swap_guide_video

    ref = tmp_path / "ref.mp4"
    face = tmp_path / "face.png"
    ref.write_bytes(b"mp4")
    face.write_bytes(b"png")

    layout_obj = type(
        "Layout",
        (),
        {
            "region_size_px": 200,
            "region_position": "left",
            "video_x": 200,
            "video_y": 0,
            "video_w": 280,
            "video_h": 704,
            "frame_w": 480,
            "frame_h": 704,
        },
    )()

    with patch("ltx_media.probe_video_info") as probe, patch(
        "ltx_face_swap_compose.resolve_face_swap_canvas_size",
        return_value=(768, 512),
    ), patch(
        "ltx_media.trim_video_fit_aspect",
        return_value=(tmp_path / "trimmed.mp4", 512, 288),
    ) as trim, patch("ltx_face_swap_compose.compose_bfs_v3_guide_video") as compose, patch(
        "ltx_face_swap_compose.compute_bfs_guide_layout",
    ) as layout_fn:
        probe.return_value = type(
            "Info", (), {"num_frames": 889, "fps": 30.0, "width": 1920, "height": 1080}
        )()
        trim.side_effect = lambda _src, dst, **_: Path(dst).write_bytes(b"trimmed") or Path(dst)
        layout_fn.return_value = layout_obj
        compose.return_value = layout_obj

        guide_path, layout, effective_nf, canvas_w, canvas_h = _prepare_face_swap_guide_video(
            str(ref),
            str(face),
            tmpdir=str(tmp_path),
            num_frames=121,
            width=480,
            height=704,
            fps=24.0,
        )

    trim.assert_called_once()
    layout_fn.assert_called_once()
    compose.assert_called_once()
    assert compose.call_args.kwargs["width"] == canvas_w
    assert compose.call_args.kwargs["height"] == canvas_h
    assert guide_path.endswith("face_swap_bfs_v3_guide.mp4")
    assert layout is layout_obj
    assert effective_nf == 121
    assert canvas_w == 768 and canvas_h == 512


def test_canvas_from_video_aspect_preserves_16_9():
    from ltx_media import canvas_from_video_aspect

    w, h = canvas_from_video_aspect(1920, 1080, 768)
    assert w == 768
    assert h == 448  # 432 snapped to nearest multiple of 32
    assert w % 32 == 0 and h % 32 == 0


def test_ic_lora_vae_compatible_frame_count():
    from ltx_media import ic_lora_vae_compatible_frame_count

    assert ic_lora_vae_compatible_frame_count(121) == 121
    assert ic_lora_vae_compatible_frame_count(121, source_num_frames=2) == 9
    assert ic_lora_vae_compatible_frame_count(25) == 25


def test_head_swap_lora_skips_pose_preprocess():
    from ltx_mlx_backend import _needs_pose_control_preprocessing

    lora = (
        "/loras/Alissonerdx__BFS-Best-Face-Swap-Video/"
        "head_swap_v3_rank_adaptive_fro_098.safetensors",
        0.98,
    )
    with patch("ltx_mlx_backend._ic_lora_reference_downscale_factor", return_value=1):
        with patch("ltx_mlx_backend._ic_lora_uses_hdr_pipeline", return_value=False):
            assert not _needs_pose_control_preprocessing([lora], [("guide.mp4", 1.0)])


def _write_h264_mp4(path: Path, *, frames: int = 25, fps: float = 24.0, width: int = 128, height: int = 96) -> None:
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


def test_compose_bfs_v3_guide_video_writes_frames(tmp_path: Path):
    pytest.importorskip("av")
    from PIL import Image

    from ltx_face_swap_compose import compose_bfs_v3_guide_video

    src = tmp_path / "src.mp4"
    face = tmp_path / "face.png"
    out = tmp_path / "guide.mp4"
    _write_h264_mp4(src, frames=9, fps=24.0, width=320, height=240)
    Image.new("RGB", (256, 256), color=(200, 120, 80)).save(face)

    layout = compose_bfs_v3_guide_video(
        src,
        face,
        out,
        width=320,
        height=240,
        num_frames=9,
        fps=24.0,
        region_size_px=96,
    )
    assert out.is_file() and out.stat().st_size > 0
    assert layout.region_position == "left"
    assert layout.video_x == layout.region_size_px

    import av

    with av.open(str(out)) as container:
        stream = container.streams.video[0]
        frames = sum(1 for _ in container.decode(stream))
    assert frames == 9


def test_extract_bfs_guide_keyframe_images(tmp_path: Path):
    pytest.importorskip("av")
    from PIL import Image

    from ltx_face_swap_compose import compose_bfs_v3_guide_video, extract_bfs_guide_keyframe_images

    src = tmp_path / "src.mp4"
    face = tmp_path / "face.png"
    guide = tmp_path / "guide.mp4"
    kf_dir = tmp_path / "keyframes"
    _write_h264_mp4(src, frames=25, fps=24.0, width=320, height=240)
    Image.new("RGB", (256, 256), color=(200, 120, 80)).save(face)
    compose_bfs_v3_guide_video(
        src,
        face,
        guide,
        width=320,
        height=240,
        num_frames=25,
        fps=24.0,
        region_size_px=96,
    )

    keyframes = extract_bfs_guide_keyframe_images(
        guide,
        kf_dir,
        num_frames=25,
        interval=8,
    )
    assert keyframes[0][1] == 0
    assert all(Path(p).is_file() for p, *_ in keyframes)
    assert len(keyframes) >= 3
    assert keyframes[-1][1] == 24


def test_extract_bfs_guide_keyframe_at_index(tmp_path: Path):
    pytest.importorskip("av")
    from PIL import Image

    from ltx_face_swap_compose import compose_bfs_v3_guide_video, extract_bfs_guide_keyframe_at_index
    from tests.test_face_swap_backend import _write_h264_mp4

    src = tmp_path / "src.mp4"
    face = tmp_path / "face.png"
    guide = tmp_path / "guide.mp4"
    kf_dir = tmp_path / "keyframes"
    _write_h264_mp4(src, frames=9, fps=24.0, width=320, height=240)
    Image.new("RGB", (256, 256), color=(200, 120, 80)).save(face)
    compose_bfs_v3_guide_video(
        src, face, guide, width=320, height=240, num_frames=9, fps=24.0, region_size_px=96,
    )
    path, idx = extract_bfs_guide_keyframe_at_index(guide, kf_dir, frame_idx=0)
    assert idx == 0
    assert Path(path).is_file()


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
