"""IC-LoRA control-signal preprocessing (pose maps for motion transfer)."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

POSE_LANDMARKER_LITE_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

# BlazePose 33-landmark topology (legacy mp.solutions.pose.POSE_CONNECTIONS).
POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
)


def _pose_model_cache_path() -> Path:
    from ltx_paths import REPO_ROOT

    root = REPO_ROOT / "models" / "mediapipe"
    root.mkdir(parents=True, exist_ok=True)
    return root / "pose_landmarker_lite.task"


def _ensure_pose_landmarker_model() -> Path:
    path = _pose_model_cache_path()
    if path.is_file() and path.stat().st_size > 0:
        return path
    log.info("Downloading MediaPipe pose landmarker model to %s", path)
    tmp = path.with_suffix(".task.download")
    urllib.request.urlretrieve(POSE_LANDMARKER_LITE_URL, tmp)
    tmp.replace(path)
    return path


def _mediapipe_has_legacy_solutions() -> bool:
    try:
        import mediapipe as mp

        return hasattr(mp, "solutions") and hasattr(mp.solutions, "pose")
    except ImportError:
        return False


def _mediapipe_has_tasks_pose() -> bool:
    try:
        from mediapipe.tasks.python import vision

        return hasattr(vision, "PoseLandmarker")
    except ImportError:
        return False


def pose_control_available() -> bool:
    try:
        import mediapipe  # noqa: F401
    except ImportError:
        return False
    return _mediapipe_has_legacy_solutions() or _mediapipe_has_tasks_pose()


def require_pose_control() -> None:
    if not pose_control_available():
        raise RuntimeError(
            "Motion-transfer IC-LoRA requires mediapipe — install with: pip install mediapipe"
        )


def _landmark_point(lm: object, *, width: int, height: int) -> tuple[int, int] | None:
    vis = getattr(lm, "visibility", None)
    if vis is not None and float(vis) < 0.5:
        return None
    pres = getattr(lm, "presence", None)
    if pres is not None and float(pres) < 0.5:
        return None
    return (int(float(lm.x) * width), int(float(lm.y) * height))


def _draw_pose_skeleton(
    canvas: "object",
    landmarks: list[object] | None,
    *,
    width: int,
    height: int,
) -> None:
    """Draw white stick-figure pose lines on a black RGB uint8 canvas."""
    if not landmarks:
        return

    pts = [_landmark_point(lm, width=width, height=height) for lm in landmarks]
    for i, j in POSE_CONNECTIONS:
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


def _pad_frame(canvas: "object", *, width: int, height: int, pad_w: int, pad_h: int) -> "object":
    import numpy as np

    if (pad_w, pad_h) == (width, height):
        return canvas
    padded = np.zeros((pad_h, pad_w, 3), dtype=np.uint8)
    padded[:height, :width, :] = canvas
    return padded


def _decode_motion_frames(
    source_video: Path,
    *,
    width: int,
    height: int,
    num_frames: int,
) -> list["object"]:
    import numpy as np

    from ltx_media import require_media

    require_media()
    import av

    frames_rgb: list[np.ndarray] = []
    with av.open(str(source_video)) as container:
        if not container.streams.video:
            raise RuntimeError(f"No video stream in {source_video}")
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            if len(frames_rgb) >= num_frames:
                break
            rgb = frame.reformat(width=width, height=height, format="rgb24")
            frames_rgb.append(np.asarray(rgb.to_ndarray(), dtype=np.uint8))
    if not frames_rgb:
        raise RuntimeError(f"No frames decoded from {source_video}")
    while len(frames_rgb) < num_frames:
        frames_rgb.append(frames_rgb[-1].copy())
    return frames_rgb[:num_frames]


def _pose_frames_legacy(
    frames_rgb: list["object"],
    *,
    width: int,
    height: int,
) -> list["object"]:
    import numpy as np
    import mediapipe as mp

    out: list[np.ndarray] = []
    last_landmarks: list[object] | None = None
    with mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        for arr in frames_rgb:
            result = pose.process(arr)
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            landmarks_list = None
            if result.pose_landmarks is not None:
                landmarks_list = list(result.pose_landmarks.landmark)
                last_landmarks = landmarks_list
            elif last_landmarks is not None:
                landmarks_list = last_landmarks
            _draw_pose_skeleton(canvas, landmarks_list, width=width, height=height)
            out.append(canvas)
    return out


def _pose_frames_tasks(
    frames_rgb: list["object"],
    *,
    width: int,
    height: int,
    fps: float,
) -> list["object"]:
    import numpy as np
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

    model_path = str(_ensure_pose_landmarker_model())
    options = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    out: list[np.ndarray] = []
    last_landmarks: list[object] | None = None
    frame_ms = 1000.0 / max(fps, 1.0)

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for idx, arr in enumerate(frames_rgb):
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
            result = landmarker.detect_for_video(mp_image, int(idx * frame_ms))
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            landmarks_list = None
            if result.pose_landmarks:
                landmarks_list = list(result.pose_landmarks[0])
                last_landmarks = landmarks_list
            elif last_landmarks is not None:
                landmarks_list = last_landmarks
            _draw_pose_skeleton(canvas, landmarks_list, width=width, height=height)
            out.append(canvas)
    return out


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

    frames_rgb = _decode_motion_frames(
        source_video, width=width, height=height, num_frames=num_frames
    )
    if _mediapipe_has_tasks_pose():
        pose_frames = _pose_frames_tasks(
            frames_rgb, width=width, height=height, fps=float(fps)
        )
    elif _mediapipe_has_legacy_solutions():
        pose_frames = _pose_frames_legacy(frames_rgb, width=width, height=height)
    else:
        raise RuntimeError(
            "Installed mediapipe has no pose API (legacy solutions or tasks). "
            "Upgrade/downgrade mediapipe or reinstall: pip install -U mediapipe"
        )

    with av.open(str(output_path), "w") as out:
        vstream = out.add_stream("libx264", rate=fps_frac, width=pad_w, height=pad_h)
        vstream.pix_fmt = "yuv420p"
        vstream.options = {"crf": "18", "preset": "veryfast"}
        for i, canvas in enumerate(pose_frames):
            arr = _pad_frame(canvas, width=width, height=height, pad_w=pad_w, pad_h=pad_h)
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
