"""IC-LoRA control-signal preprocessing (pose maps for motion transfer)."""

from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger(__name__)


def pose_control_available() -> bool:
    try:
        import mediapipe  # noqa: F401

        return True
    except ImportError:
        return False


def require_pose_control() -> None:
    if not pose_control_available():
        raise RuntimeError(
            "Motion-transfer IC-LoRA requires mediapipe — install with: pip install mediapipe"
        )


def _draw_pose_skeleton(
    canvas: "object",
    landmarks: "object",
    *,
    width: int,
    height: int,
) -> None:
    """Draw white stick-figure pose lines on a black RGB uint8 canvas."""
    import numpy as np
    import mediapipe as mp

    if landmarks is None:
        return

    pts: list[tuple[int, int] | None] = []
    for lm in landmarks.landmark:
        if lm.visibility < 0.5:
            pts.append(None)
            continue
        x = int(lm.x * width)
        y = int(lm.y * height)
        pts.append((x, y))

    for i, j in mp.solutions.pose.POSE_CONNECTIONS:
        a = pts[i] if i < len(pts) else None
        b = pts[j] if j < len(pts) else None
        if a is None or b is None:
            continue
        _bresenham_line(canvas, a, b)

    for p in pts:
        if p is None:
            continue
        x, y = p
        y0, y1 = max(0, y - 2), min(height, y + 3)
        x0, x1 = max(0, x - 2), min(width, x + 3)
        canvas[y0:y1, x0:x1] = 255


def _bresenham_line(canvas: "object", a: tuple[int, int], b: tuple[int, int]) -> None:
    import numpy as np

    x0, y0 = a
    x1, y1 = b
    h, w = canvas.shape[:2]
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        if 0 <= x0 < w and 0 <= y0 < h:
            canvas[y0, x0] = 255
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def render_pose_control_video(
    source_video: str | Path,
    output_path: str | Path,
    *,
    width: int,
    height: int,
    num_frames: int,
    fps: float = 24.0,
) -> Path:
    """Render per-frame pose stick figures (Union Control signal) from a motion clip."""
    import numpy as np
    import mediapipe as mp

    from ltx_media import _pyav_frame_rate, require_media

    require_pose_control()
    require_media()
    import av

    source_video = Path(source_video)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    width = int(width)
    height = int(height)
    num_frames = max(1, int(num_frames))
    fps_frac = _pyav_frame_rate(fps)

    pad_w = width + (width & 1)
    pad_h = height + (height & 1)

    frames_rgb: list[np.ndarray] = []
    last_landmarks = None

    with mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        with av.open(str(source_video)) as container:
            if not container.streams.video:
                raise RuntimeError(f"No video stream in {source_video}")
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                if len(frames_rgb) >= num_frames:
                    break
                rgb = frame.reformat(width=width, height=height, format="rgb24")
                arr = np.asarray(rgb.to_ndarray(), dtype=np.uint8)
                result = pose.process(arr)
                canvas = np.zeros((height, width, 3), dtype=np.uint8)
                landmarks = result.pose_landmarks
                if landmarks is not None:
                    last_landmarks = landmarks
                elif last_landmarks is not None:
                    landmarks = last_landmarks
                _draw_pose_skeleton(canvas, landmarks, width=width, height=height)
                if (pad_w, pad_h) != (width, height):
                    padded = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
                    padded[:height, :width, :] = canvas
                    canvas = padded
                frames_rgb.append(canvas)

    if not frames_rgb:
        raise RuntimeError(f"No frames decoded from {source_video}")

    while len(frames_rgb) < num_frames:
        frames_rgb.append(frames_rgb[-1].copy())

    with av.open(str(output_path), "w") as out:
        vstream = out.add_stream("libx264", rate=fps_frac, width=pad_w, height=pad_h)
        vstream.pix_fmt = "yuv420p"
        vstream.options = {"crf": "18", "preset": "veryfast"}
        for i, arr in enumerate(frames_rgb[:num_frames]):
            vf = av.VideoFrame.from_ndarray(arr, format="rgb24")
            vf = vf.reformat(format="yuv420p")
            vf.pts = i
            for packet in vstream.encode(vf):
                out.mux(packet)
        for packet in vstream.encode(None):
            out.mux(packet)

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError(f"Pose control encode produced empty output: {output_path}")
    log.info(
        "IC-LoRA pose control: %s → %s (%d frames, %dx%d)",
        source_video,
        output_path,
        num_frames,
        width,
        height,
    )
    return output_path
