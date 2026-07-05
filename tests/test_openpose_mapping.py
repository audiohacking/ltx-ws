"""OpenPose mapping for IC-LoRA pose control."""

from __future__ import annotations

from types import SimpleNamespace

from ltx_ic_lora_preprocess import mediapipe_landmarks_to_openpose18


def _lm(x: float, y: float, v: float = 0.9) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, visibility=v, presence=v)


def test_mediapipe_to_openpose18_neck_and_nose():
    landmarks = [_lm(0, 0)] * 33
    landmarks[0] = _lm(0.5, 0.2)  # nose
    landmarks[11] = _lm(0.4, 0.4)  # left shoulder
    landmarks[12] = _lm(0.6, 0.4)  # right shoulder
    landmarks[16] = _lm(0.7, 0.55)  # right wrist

    pts = mediapipe_landmarks_to_openpose18(landmarks, width=100, height=200)
    assert pts[0] == (50, 40)
    assert pts[1] == (50, 80)  # neck midpoint
    assert pts[4] == (70, 110)  # right wrist
