"""Face swap pipeline wiring."""

from __future__ import annotations


def test_resolve_face_swap_canvas_exports():
    from ltx_face_swap_compose import FACE_SWAP_DEFAULT_LONGER_EDGE, resolve_face_swap_canvas_size

    assert FACE_SWAP_DEFAULT_LONGER_EDGE == 768
    w, h = resolve_face_swap_canvas_size(1920, 1080, request_width=704, request_height=480)
    assert w >= h
    assert w % 32 == 0 and h % 32 == 0


def test_face_swap_pipeline_class_exports():
    from ltx_face_swap_pipeline import FaceSwapPipeline
    from ltx_pipelines_mlx.ic_lora import ICLoraPipeline

    assert issubclass(FaceSwapPipeline, ICLoraPipeline)
    assert hasattr(FaceSwapPipeline, "generate_face_swap")
