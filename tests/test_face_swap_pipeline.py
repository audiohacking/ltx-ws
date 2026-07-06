"""Face swap pipeline wiring."""

from __future__ import annotations


def test_resolve_face_swap_canvas_exports():
    from ltx_face_swap_compose import FACE_SWAP_DEFAULT_LONGER_EDGE, resolve_face_swap_canvas_size

    assert FACE_SWAP_DEFAULT_LONGER_EDGE == 768
    w, h = resolve_face_swap_canvas_size(1920, 1080, request_width=704, request_height=480)
    assert w >= h
    assert w % 32 == 0 and h % 32 == 0


def test_face_swap_pipeline_uses_dev_cfg_keyframes_in_source():
    from pathlib import Path

    src = Path("ltx_face_swap_pipeline.py").read_text(encoding="utf-8")
    assert "KeyframeInterpolationPipeline" in src
    assert "extract_bfs_guide_keyframe_images" in src
    assert "append_ic_lora_reference_video_conditionings" not in src
    assert "_encode_guide_video_tokens" not in src


def test_face_swap_pipeline_class_exports():
    from ltx_face_swap_pipeline import (
        DEFAULT_FACE_SWAP_CFG,
        DEFAULT_FACE_SWAP_STAGE1_STEPS,
        FaceSwapPipeline,
    )
    from ltx_pipelines_mlx.keyframe_interpolation import KeyframeInterpolationPipeline

    assert issubclass(FaceSwapPipeline, KeyframeInterpolationPipeline)
    assert hasattr(FaceSwapPipeline, "generate_face_swap")
    assert DEFAULT_FACE_SWAP_CFG == 3.0
    assert DEFAULT_FACE_SWAP_STAGE1_STEPS == 20
