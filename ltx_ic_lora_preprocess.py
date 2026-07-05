"""IC-LoRA control-signal preprocessing (ComfyUI OpenPose maps for Union Control)."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

log = logging.getLogger(__name__)

POSE_LANDMARKER_FULL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
)

# ComfyUI controlnet_aux ``draw_bodypose`` limb sequence (1-based → 0-based indices).
OPENPOSE_BODY18_LIMBS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (0, 1),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
)

# Same 18-color ramp as ComfyUI ``draw_bodypose`` (RGB).
OPENPOSE_LIMB_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 0, 0),
    (255, 85, 0),
    (255, 170, 0),
    (255, 255, 0),
    (170, 255, 0),
    (85, 255, 0),
    (0, 255, 0),
    (0, 255, 85),
    (0, 255, 170),
    (0, 255, 255),
    (0, 170, 255),
    (0, 85, 255),
    (0, 0, 255),
    (85, 0, 255),
    (170, 0, 255),
    (255, 0, 255),
    (255, 0, 170),
    (255, 0, 85),
)

# MediaPipe Pose → OpenPose BODY-18 (ComfyUI / DWPreprocessor topology).
_MP_TO_OPENPOSE: dict[int, int] = {
    0: 0,  # nose
    2: 15,  # left eye
    5: 14,  # right eye
    7: 17,  # left ear
    8: 16,  # right ear
    12: 2,  # right shoulder
    14: 3,  # right elbow
    16: 4,  # right wrist
    11: 5,  # left shoulder
    13: 6,  # left elbow
    15: 7,  # left wrist
    24: 8,  # right hip
    26: 9,  # right knee
    28: 10,  # right ankle
    23: 11,  # left hip
    25: 12,  # left knee
    27: 13,  # left ankle
}


def _pose_model_cache_path() -> Path:
    from ltx_paths import REPO_ROOT

    root = REPO_ROOT / "models" / "mediapipe"
    root.mkdir(parents=True, exist_ok=True)
    return root / "pose_landmarker_full.task"


def _ensure_pose_landmarker_model() -> Path:
    path = _pose_model_cache_path()
    if path.is_file() and path.stat().st_size > 0:
        return path
    log.info("Downloading MediaPipe pose landmarker (full) to %s", path)
    tmp = path.with_suffix(".task.download")
    urllib.request.urlretrieve(POSE_LANDMARKER_FULL_URL, tmp)
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


def _mp_point(
    landmarks: list[object],
    idx: int,
    *,
    width: int,
    height: int,
    min_score: float = 0.35,
) -> tuple[float, float] | None:
    if idx >= len(landmarks):
        return None
    lm = landmarks[idx]
    vis = getattr(lm, "visibility", None)
    if vis is not None and float(vis) < min_score:
        return None
    pres = getattr(lm, "presence", None)
    if pres is not None and float(pres) < min_score:
        return None
    return (float(lm.x) * width, float(lm.y) * height)


def mediapipe_landmarks_to_openpose18(
    landmarks: list[object] | None,
    *,
    width: int,
    height: int,
) -> list[tuple[int, int] | None]:
    """Map MediaPipe Pose landmarks to OpenPose BODY-18 pixel keypoints."""
    pts: list[tuple[int, int] | None] = [None] * 18
    if not landmarks:
        return pts

    for mp_idx, op_idx in _MP_TO_OPENPOSE.items():
        p = _mp_point(landmarks, mp_idx, width=width, height=height)
        if p is not None:
            pts[op_idx] = (int(p[0]), int(p[1]))

    l_sh = _mp_point(landmarks, 11, width=width, height=height)
    r_sh = _mp_point(landmarks, 12, width=width, height=height)
    if l_sh is not None and r_sh is not None:
        pts[1] = (int((l_sh[0] + r_sh[0]) / 2), int((l_sh[1] + r_sh[1]) / 2))

    return pts


def _draw_thick_colored_line(
    canvas: "object",
    a: tuple[int, int],
    b: tuple[int, int],
    color: tuple[int, int, int],
    *,
    thickness: int,
) -> None:
    import numpy as np

    x0, y0 = a
    x1, y1 = b
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    rgb = np.array(color, dtype=np.uint8)
    for t in np.linspace(0.0, 1.0, num=steps + 1):
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        y0b, y1b = max(0, y - thickness), min(canvas.shape[0], y + thickness + 1)
        x0b, x1b = max(0, x - thickness), min(canvas.shape[1], x + thickness + 1)
        canvas[y0b:y1b, x0b:x1b] = rgb


def _draw_openpose_skeleton(
    canvas: "object",
    openpose_pts: list[tuple[int, int] | None],
    *,
    line_thickness: int = 4,
    joint_radius: int = 4,
) -> None:
    """Render ComfyUI-style colored OpenPose BODY-18 on black (Union Control training format)."""
    import numpy as np

    h, w = canvas.shape[:2]
    for (i, j), color in zip(OPENPOSE_BODY18_LIMBS, OPENPOSE_LIMB_COLORS, strict=False):
        a = openpose_pts[i] if i < len(openpose_pts) else None
        b = openpose_pts[j] if j < len(openpose_pts) else None
        if a is None or b is None:
            continue
        limb_color = tuple(int(float(c) * 0.6) for c in color)
        _draw_thick_colored_line(canvas, a, b, limb_color, thickness=line_thickness)

    for idx, p in enumerate(openpose_pts):
        if p is None:
            continue
        color = OPENPOSE_LIMB_COLORS[idx % len(OPENPOSE_LIMB_COLORS)]
        x, y = p
        y0, y1 = max(0, y - joint_radius), min(h, y + joint_radius + 1)
        x0, x1 = max(0, x - joint_radius), min(w, x + joint_radius + 1)
        canvas[y0:y1, x0:x1] = np.array(color, dtype=np.uint8)


def _pose_frame_energy(frame: "object") -> int:
    import numpy as np

    arr = np.asarray(frame)
    if arr.ndim == 3:
        return int((arr.max(axis=2) > 16).sum())
    return int((arr > 16).sum())


def _draw_pose_from_landmarks(
    canvas: "object",
    landmarks: list[object] | None,
    *,
    width: int,
    height: int,
) -> None:
    openpose_pts = mediapipe_landmarks_to_openpose18(landmarks, width=width, height=height)
    _draw_openpose_skeleton(canvas, openpose_pts)


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
    fps: float,
) -> list["object"]:
    import numpy as np

    from ltx_media import require_media

    require_media()
    import av

    frames_rgb: list[np.ndarray] = []
    next_pick = 0.0
    decoded_index = 0
    target_fps = max(float(fps), 1.0)

    with av.open(str(source_video)) as container:
        if not container.streams.video:
            raise RuntimeError(f"No video stream in {source_video}")
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate or stream.base_rate or target_fps)

        for frame in container.decode(stream):
            if len(frames_rgb) >= num_frames:
                break

            take = True
            if source_fps > 0 and abs(source_fps - target_fps) > 0.01:
                take = decoded_index >= next_pick
                if take:
                    next_pick += source_fps / target_fps

            if not take:
                decoded_index += 1
                continue

            rgb = frame.reformat(width=width, height=height, format="rgb24")
            frames_rgb.append(np.ascontiguousarray(rgb.to_ndarray(), dtype=np.uint8))
            decoded_index += 1

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
) -> tuple[list["object"], int]:
    import numpy as np
    import mediapipe as mp

    out: list[np.ndarray] = []
    last_landmarks: list[object] | None = None
    detected = 0
    with mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.35,
        min_tracking_confidence=0.35,
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
            _draw_pose_from_landmarks(canvas, landmarks_list, width=width, height=height)
            if _pose_frame_energy(canvas) > 500:
                detected += 1
            out.append(canvas)
    return out, detected


def _pose_frames_tasks(
    frames_rgb: list["object"],
    *,
    width: int,
    height: int,
    fps: float,
) -> tuple[list["object"], int]:
    import numpy as np
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision

    model_path = str(_ensure_pose_landmarker_model())
    options = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.35,
        min_pose_presence_confidence=0.35,
        min_tracking_confidence=0.35,
    )
    out: list[np.ndarray] = []
    last_landmarks: list[object] | None = None
    detected = 0
    frame_ms = 1000.0 / max(fps, 1.0)

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for idx, arr in enumerate(frames_rgb):
            arr = np.ascontiguousarray(arr, dtype=np.uint8)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
            timestamp_ms = int(round(idx * frame_ms))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            landmarks_list = None
            if result.pose_landmarks:
                landmarks_list = list(result.pose_landmarks[0])
                last_landmarks = landmarks_list
            elif last_landmarks is not None:
                landmarks_list = last_landmarks
            _draw_pose_from_landmarks(canvas, landmarks_list, width=width, height=height)
            if _pose_frame_energy(canvas) > 500:
                detected += 1
            out.append(canvas)
    return out, detected


def render_pose_control_video(
    source_video: str | Path,
    output_path: str | Path,
    *,
    width: int,
    height: int,
    num_frames: int,
    fps: float = 24.0,
) -> Path:
    """Render ComfyUI-style colored OpenPose maps for Union Control IC-LoRA."""
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
        source_video,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=float(fps),
    )
    if _mediapipe_has_tasks_pose():
        pose_frames, detected = _pose_frames_tasks(
            frames_rgb, width=width, height=height, fps=float(fps)
        )
    elif _mediapipe_has_legacy_solutions():
        pose_frames, detected = _pose_frames_legacy(
            frames_rgb, width=width, height=height
        )
    else:
        raise RuntimeError(
            "Installed mediapipe has no pose API (legacy solutions or tasks). "
            "Upgrade/downgrade mediapipe or reinstall: pip install -U mediapipe"
        )

    min_detected = max(3, int(len(pose_frames) * 0.15))
    if detected < min_detected:
        raise RuntimeError(
            f"Pose extraction found bodies in only {detected}/{len(pose_frames)} frames "
            f"(need >={min_detected}). Use a clearer full-body motion reference, or "
            "motion-only mode (video without character image) with HDR IC-LoRA."
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
        "IC-LoRA OpenPose control: %s → %s (%d frames, %dx%d, detected=%d/%d, colored=ComfyUI)",
        source_video,
        output_path,
        num_frames,
        width,
        height,
        detected,
        len(pose_frames),
    )
    return output_path
