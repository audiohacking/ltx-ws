"""BFS head-swap V3 guide-video composition (Reserved Region Frame Composer).

The Alissonerdx ``head_swap_v3_*`` LoRA is trained on composite guide clips:
performance video in the main area plus a persistent identity face in a green
chroma side panel. IC-LoRA ``video_conditioning`` must receive this composite,
not raw reference footage with a separate ``images`` anchor.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import logging

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceSwapGuideLayout:
    """Pixel layout of the BFS V3 composite relative to the output frame."""

    region_size_px: int
    region_position: str
    video_x: int
    video_y: int
    video_w: int
    video_h: int
    frame_w: int
    frame_h: int


def _fit_inside(src_w: int, src_h: int, max_w: int, max_h: int) -> tuple[int, int]:
    if src_w <= 0 or src_h <= 0:
        return 1, 1
    scale = min(max_w / src_w, max_h / src_h)
    return max(1, int(round(src_w * scale))), max(1, int(round(src_h * scale)))


def _aligned_offset(container_size: int, content_size: int, align: str) -> int:
    if align == "start":
        return 0
    if align == "end":
        return max(0, container_size - content_size)
    return max(0, (container_size - content_size) // 2)


def _add_white_padding(img, pad_px: int = 16):
    from PIL import Image

    img = img.convert("RGBA")
    canvas = Image.new("RGBA", (img.width + pad_px * 2, img.height + pad_px * 2), (255, 255, 255, 255))
    canvas.paste(img, (pad_px, pad_px), img)
    return canvas


def _resize_face_for_region(face, *, region_w: int, region_h: int, face_scale_pct: float, face_padding_px: int):
    from PIL import Image

    usable_w = max(1, region_w - 2 * face_padding_px)
    usable_h = max(1, region_h - 2 * face_padding_px)
    target_w = max(1, int(round(usable_w * (face_scale_pct / 100.0))))
    target_h = max(1, int(round(usable_h * (face_scale_pct / 100.0))))
    tw, th = _fit_inside(face.width, face.height, target_w, target_h)
    return face.resize((tw, th), Image.Resampling.LANCZOS)


def default_region_size_px(frame_width: int) -> int:
    """Match typical BFS V3 workflows (~35% width, clamped)."""
    return max(200, min(400, int(round(frame_width * 0.35))))


def format_head_swap_prompt(user_prompt: str) -> str:
    """Wrap user text in the BFS V3 ``head_swap:`` trigger format."""
    text = (user_prompt or "").strip()
    if text.lower().startswith("head_swap:"):
        return text
    action = text or "A person performing the actions shown in the main video area."
    return (
        "head_swap:\n\n"
        "FACE:\n"
        "Use the identity from the side-panel reference face only.\n\n"
        f"ACTION:\n{action}"
    )


def compose_bfs_v3_guide_video(
    source_video: str | Path,
    identity_image: str | Path,
    output_video: str | Path,
    *,
    width: int,
    height: int,
    num_frames: int,
    fps: float,
    region_size_px: int | None = None,
    region_position: str = "left",
    chroma_rgb: tuple[int, int, int] = (0, 255, 0),
    face_scale_pct: float = 90.0,
    face_padding_px: int = 12,
) -> FaceSwapGuideLayout:
    """Build the BFS V3 composite guide clip used as IC-LoRA reference video."""
    import av
    import numpy as np
    from PIL import Image

    from ltx_media import require_media

    require_media()
    source_video = Path(source_video)
    identity_image = Path(identity_image)
    output_video = Path(output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)

    width = int(width)
    height = int(height)
    num_frames = max(1, int(num_frames))
    region_position = (region_position or "left").strip().lower()
    if region_position not in ("left", "right", "top", "bottom"):
        raise ValueError(f"Unsupported region_position: {region_position}")

    region_size_px = int(region_size_px or default_region_size_px(width))
    if region_position in ("left", "right"):
        region_size_px = max(1, min(region_size_px, width - 1))
        region_w, region_h = region_size_px, height
        video_max_w, video_max_h = width - region_size_px, height
    else:
        region_size_px = max(1, min(region_size_px, height - 1))
        region_w, region_h = width, region_size_px
        video_max_w, video_max_h = width, height - region_size_px

    fitted_video_w, fitted_video_h = _fit_inside(width, height, video_max_w, video_max_h)

    with Image.open(identity_image) as im:
        face = _add_white_padding(im.convert("RGBA"))
    face_resized = _resize_face_for_region(
        face,
        region_w=region_w,
        region_h=region_h,
        face_scale_pct=face_scale_pct,
        face_padding_px=face_padding_px,
    )

    if region_position == "left":
        region_x, region_y = 0, 0
        video_x, video_y = region_size_px, (height - fitted_video_h) // 2
    elif region_position == "right":
        region_x, region_y = width - region_size_px, 0
        video_x, video_y = 0, (height - fitted_video_h) // 2
    elif region_position == "top":
        region_x, region_y = 0, 0
        video_x, video_y = (width - fitted_video_w) // 2, region_size_px
    else:
        region_x, region_y = 0, height - region_size_px
        video_x, video_y = (width - fitted_video_w) // 2, 0

    local_x = face_padding_px + _aligned_offset(
        max(1, region_w - 2 * face_padding_px),
        face_resized.width,
        "center",
    )
    local_y = face_padding_px + _aligned_offset(
        max(1, region_h - 2 * face_padding_px),
        face_resized.height,
        "center",
    )
    face_xy = (region_x + local_x, region_y + local_y)

    chroma_rgba = (*chroma_rgb, 255)
    fps_frac = Fraction(int(round(fps * 1000)), 1000)
    pad_w = width + (width & 1)
    pad_h = height + (height & 1)
    frames_written = 0
    last_out_arr = None

    with av.open(str(source_video)) as vin, av.open(str(output_video), "w") as vout:
        if not vin.streams.video:
            raise RuntimeError(f"No video stream in {source_video}")
        in_stream = vin.streams.video[0]
        out_stream = vout.add_stream("libx264", rate=fps_frac, width=pad_w, height=pad_h)
        out_stream.pix_fmt = "yuv420p"
        out_stream.options = {"crf": "18", "preset": "veryfast"}
        out_stream.time_base = Fraction(fps_frac.denominator, fps_frac.numerator)

        for frame in vin.decode(in_stream):
            if frames_written >= num_frames:
                break
            rgb = frame.reformat(format="rgb24")
            arr = np.asarray(rgb.to_ndarray(), dtype=np.uint8)
            frame_pil = Image.fromarray(arr, mode="RGB").resize(
                (fitted_video_w, fitted_video_h),
                Image.Resampling.LANCZOS,
            )
            canvas = Image.new("RGBA", (width, height), (0, 0, 0, 255))
            region_img = Image.new("RGBA", (region_w, region_h), chroma_rgba)
            canvas.paste(region_img, (region_x, region_y))
            canvas.paste(frame_pil.convert("RGBA"), (video_x, video_y))
            canvas.paste(face_resized, face_xy, face_resized)
            out_arr = np.asarray(canvas.convert("RGB"), dtype=np.uint8)
            if pad_w != width or pad_h != height:
                padded = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
                padded[:height, :width, :] = out_arr
                out_arr = padded
            last_out_arr = out_arr
            out_frame = av.VideoFrame.from_ndarray(out_arr, format="rgb24")
            out_frame = out_frame.reformat(format="yuv420p")
            out_frame.pts = frames_written
            for packet in out_stream.encode(out_frame):
                vout.mux(packet)
            frames_written += 1

        if frames_written == 0:
            raise RuntimeError(f"No frames composed for face-swap guide from {source_video}")

        while frames_written < num_frames and last_out_arr is not None:
            out_frame = av.VideoFrame.from_ndarray(last_out_arr, format="rgb24")
            out_frame = out_frame.reformat(format="yuv420p")
            out_frame.pts = frames_written
            for packet in out_stream.encode(out_frame):
                vout.mux(packet)
            frames_written += 1

        for packet in out_stream.encode(None):
            vout.mux(packet)

    layout = FaceSwapGuideLayout(
        region_size_px=region_size_px,
        region_position=region_position,
        video_x=video_x,
        video_y=video_y,
        video_w=fitted_video_w,
        video_h=fitted_video_h,
        frame_w=width,
        frame_h=height,
    )
    log.info(
        "Face swap: BFS V3 composite guide %dx%d (%d frames) region=%s %dpx video@%d,%d",
        width,
        height,
        frames_written,
        region_position,
        region_size_px,
        video_x,
        video_y,
    )
    return layout


def crop_face_swap_output_to_main_video(
    video_path: str | Path,
    layout: FaceSwapGuideLayout,
) -> Path:
    """Crop generated composite output back to the main performance area."""
    import shutil
    import tempfile

    import av
    import numpy as np
    from PIL import Image

    from ltx_media import require_media

    require_media()
    video_path = Path(video_path)
    crop_box = (
        layout.video_x,
        layout.video_y,
        layout.video_x + layout.video_w,
        layout.video_y + layout.video_h,
    )
    frames_written = 0
    fd, tmp_name = tempfile.mkstemp(suffix=".mp4", prefix="ltx_face_swap_crop_")
    import os

    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        with av.open(str(video_path)) as vin, av.open(str(tmp_path), "w") as vout:
            if not vin.streams.video:
                raise RuntimeError(f"No video stream in {video_path}")
            in_stream = vin.streams.video[0]
            rate = in_stream.average_rate or in_stream.codec_context.framerate or 24
            out_w = layout.video_w + (layout.video_w & 1)
            out_h = layout.video_h + (layout.video_h & 1)
            out_stream = vout.add_stream("libx264", rate=rate, width=out_w, height=out_h)
            out_stream.pix_fmt = "yuv420p"
            out_stream.options = {"crf": "18", "preset": "veryfast"}

            for frame in vin.decode(in_stream):
                rgb = frame.reformat(format="rgb24")
                arr = np.asarray(rgb.to_ndarray(), dtype=np.uint8)
                img = Image.fromarray(arr, mode="RGB")
                cropped = img.crop(crop_box)
                if cropped.size != (layout.video_w, layout.video_h):
                    cropped = cropped.resize(
                        (layout.video_w, layout.video_h),
                        Image.Resampling.LANCZOS,
                    )
                out_arr = np.asarray(cropped, dtype=np.uint8)
                if out_w != layout.video_w or out_h != layout.video_h:
                    padded = np.zeros((out_h, out_w, 3), dtype=np.uint8)
                    padded[: layout.video_h, : layout.video_w, :] = out_arr
                    out_arr = padded
                out_frame = av.VideoFrame.from_ndarray(out_arr, format="rgb24")
                out_frame = out_frame.reformat(format="yuv420p")
                out_frame.pts = frames_written
                for packet in out_stream.encode(out_frame):
                    vout.mux(packet)
                frames_written += 1
            for packet in out_stream.encode(None):
                vout.mux(packet)

        if frames_written == 0:
            raise RuntimeError(f"Face-swap crop produced no frames from {video_path}")
        shutil.move(str(tmp_path), str(video_path))
        return video_path
    finally:
        tmp_path.unlink(missing_ok=True)
