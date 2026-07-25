"""BFS head-swap V3 guide-video composition (Reserved Region Frame Composer).

The Alissonerdx ``head_swap_v3_*`` LoRA is trained on composite guide clips:
performance video in the main area plus a persistent identity face in a green
chroma side panel. MLX inference uses IC-LoRA ``video_conditioning`` on the
composite plus a frame-0 composite ``images`` anchor for stage 2.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import logging

log = logging.getLogger(__name__)

# BFS V3 / Comfy hold interval (minimum 4; 8 matches typical IC-LoRA keyframe spacing).
DEFAULT_BFS_GUIDE_KEYFRAME_INTERVAL = 8


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
    from ltx_media import dimensions_fit_inside

    return dimensions_fit_inside(src_w, src_h, max_w, max_h)


def compute_bfs_guide_layout(
    width: int,
    height: int,
    *,
    src_width: int,
    src_height: int,
    region_size_px: int | None = None,
    region_position: str = "left",
) -> FaceSwapGuideLayout:
    """Pixel layout for BFS composite before encoding (main panel from source aspect)."""
    width = int(width)
    height = int(height)
    region_position = (region_position or "left").strip().lower()
    if region_position not in ("left", "right", "top", "bottom"):
        raise ValueError(f"Unsupported region_position: {region_position}")

    region_size_px = int(region_size_px or default_region_size_px(width))
    if region_position in ("left", "right"):
        region_size_px = max(1, min(region_size_px, width - 1))
        video_max_w, video_max_h = width - region_size_px, height
    else:
        region_size_px = max(1, min(region_size_px, height - 1))
        video_max_w, video_max_h = width, height - region_size_px

    fitted_video_w, fitted_video_h = _fit_inside(
        int(src_width),
        int(src_height),
        video_max_w,
        video_max_h,
    )

    if region_position == "left":
        video_x, video_y = region_size_px, (height - fitted_video_h) // 2
    elif region_position == "right":
        video_x, video_y = 0, (height - fitted_video_h) // 2
    elif region_position == "top":
        video_x, video_y = (width - fitted_video_w) // 2, region_size_px
    else:
        video_x, video_y = (width - fitted_video_w) // 2, 0

    return FaceSwapGuideLayout(
        region_size_px=region_size_px,
        region_position=region_position,
        video_x=video_x,
        video_y=video_y,
        video_w=fitted_video_w,
        video_h=fitted_video_h,
        frame_w=width,
        frame_h=height,
    )


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


# BFS V3 recommends ~768px; Comfy workflow defaults to landscape 768×512 class sizes.
FACE_SWAP_DEFAULT_LONGER_EDGE = 768


def resolve_face_swap_canvas_size(
    src_width: int,
    src_height: int,
    *,
    request_width: int | None = None,
    request_height: int | None = None,
) -> tuple[int, int]:
    """Pick generation canvas from source aspect (no stretch)."""
    from ltx_media import canvas_from_video_aspect

    longer = FACE_SWAP_DEFAULT_LONGER_EDGE
    if request_width and request_height:
        longer = max(int(request_width), int(request_height), 512)
    elif request_width:
        longer = max(int(request_width), 512)
    elif request_height:
        longer = max(int(request_height), 512)
    return canvas_from_video_aspect(src_width, src_height, longer)


def default_region_size_px(frame_width: int) -> int:
    """Match BFS V3 ComfyUI workflow default (256px side strip)."""
    return max(200, min(400, 256))


def format_head_swap_prompt(user_prompt: str) -> str:
    """Wrap user text in the BFS V3 ``head_swap:`` trigger format.

    Identity comes from the side-panel reference image in the composite guide;
    the prompt only needs the ``head_swap:`` trigger and an ``ACTION`` line.
    """
    text = (user_prompt or "").strip()
    if text.lower().startswith("head_swap:"):
        return text
    action = text or "Perform the actions shown in the main video area."
    return f"head_swap:\n\nFACE:\n\nACTION:\n{action}"


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
    face_scale_pct: float = 100.0,
    face_padding_px: int = 12,
    layout: FaceSwapGuideLayout | None = None,
    src_width: int | None = None,
    src_height: int | None = None,
) -> FaceSwapGuideLayout:
    """Build the BFS V3 composite guide clip (Comfy ReservedRegionFrameComposer)."""
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

    if layout is None:
        if src_width is None or src_height is None:
            with av.open(str(source_video)) as peek:
                if not peek.streams.video:
                    raise RuntimeError(f"No video stream in {source_video}")
                first = next(peek.decode(peek.streams.video[0]))
                rgb = first.reformat(format="rgb24")
                arr = np.asarray(rgb.to_ndarray(), dtype=np.uint8)
                src_height, src_width = arr.shape[:2]
        layout = compute_bfs_guide_layout(
            width,
            height,
            src_width=int(src_width),
            src_height=int(src_height),
            region_size_px=region_size_px,
            region_position=region_position,
        )

    region_size_px = layout.region_size_px
    region_position = layout.region_position
    fitted_video_w, fitted_video_h = layout.video_w, layout.video_h
    video_x, video_y = layout.video_x, layout.video_y

    if region_position in ("left", "right"):
        region_w, region_h = region_size_px, height
        region_x = 0 if region_position == "left" else width - region_size_px
        region_y = 0
    else:
        region_w, region_h = width, region_size_px
        region_x = 0
        region_y = 0 if region_position == "top" else height - region_size_px

    with Image.open(identity_image) as im:
        face = _add_white_padding(im.convert("RGBA"))
    face_resized = _resize_face_for_region(
        face,
        region_w=region_w,
        region_h=region_h,
        face_scale_pct=face_scale_pct,
        face_padding_px=face_padding_px,
    )

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
            frame_pil = Image.fromarray(arr, mode="RGB")
            if frame_pil.size != (fitted_video_w, fitted_video_h):
                frame_pil = frame_pil.resize(
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


def extract_bfs_guide_keyframe_images(
    guide_video: str | Path,
    tmpdir: str | Path,
    *,
    num_frames: int,
    interval: int = DEFAULT_BFS_GUIDE_KEYFRAME_INTERVAL,
    strength: float = 1.0,
    crf: int = 33,
) -> list[tuple[str, int, float, int]]:
    """Sample composite guide frames for AddGuideMulti-style keyframe conditioning."""
    import av
    import numpy as np
    from PIL import Image

    guide_video = Path(guide_video)
    tmpdir = Path(tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)

    interval = max(1, int(interval))
    num_frames = max(1, int(num_frames))
    target_indices = list(range(0, num_frames, interval))
    if num_frames > 1 and (num_frames - 1) not in target_indices:
        target_indices.append(num_frames - 1)
    indices_set = set(target_indices)

    extracted: list[tuple[str, int, float, int]] = []
    with av.open(str(guide_video)) as container:
        if not container.streams.video:
            raise RuntimeError(f"No video stream in {guide_video}")
        stream = container.streams.video[0]
        for frame_idx, frame in enumerate(container.decode(stream)):
            if frame_idx >= num_frames:
                break
            if frame_idx not in indices_set:
                continue
            rgb = frame.reformat(format="rgb24")
            arr = np.asarray(rgb.to_ndarray(), dtype=np.uint8)
            out_path = tmpdir / f"bfs_guide_kf_{frame_idx:05d}.png"
            Image.fromarray(arr, mode="RGB").save(out_path)
            extracted.append((str(out_path), frame_idx, float(strength), int(crf)))

    if not extracted:
        raise RuntimeError(f"No keyframes extracted from BFS guide video {guide_video}")

    log.info(
        "Face swap: extracted %d composite guide keyframes (interval=%d) "
        "for AddGuideMulti-style conditioning",
        len(extracted),
        interval,
    )
    return extracted


def extract_bfs_guide_keyframe_at_index(
    guide_video: str | Path,
    tmpdir: str | Path,
    *,
    frame_idx: int = 0,
    crf: int = 33,
) -> tuple[str, int]:
    """Extract one composite guide frame for ``LTXVAddGuideMulti`` (BFS V3 uses frame 0)."""
    items = extract_bfs_guide_keyframe_images(
        guide_video,
        tmpdir,
        num_frames=max(1, int(frame_idx) + 1),
        interval=max(1, int(frame_idx) + 1),
        crf=crf,
    )
    for path, idx, *_ in items:
        if idx == frame_idx:
            return path, idx
    return items[0][0], items[0][1]


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
