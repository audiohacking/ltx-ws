"""Ensure I2V image preprocess uses PyAV instead of system ffmpeg."""

from __future__ import annotations

import sys
from io import BytesIO

import numpy as np
import pytest


def test_single_frame_h264_roundtrip():
    pytest.importorskip("av")
    from ltx_media import decode_single_frame, encode_single_frame

    img = np.zeros((101, 102, 3), dtype=np.uint8)
    img[10:90, 10:90] = 255
    buf = BytesIO()
    encode_single_frame(buf, img, crf=33)
    out = decode_single_frame(buf)
    assert out.shape[0] >= 101
    assert out.shape[1] >= 102


def test_media_io_uses_pyav_patch():
    pytest.importorskip("ltx_pipelines_mlx")
    from ltx_pipelines_mlx.utils import media_io as media_mod
    from ltx_mlx_backend import _patch_media_io_pyav_only
    import ltx_media

    _patch_media_io_pyav_only()
    assert getattr(media_mod, "_ltx_ws_pyav_media_patched", False)
    assert media_mod.encode_single_frame is ltx_media.encode_single_frame
    assert media_mod.decode_single_frame is ltx_media.decode_single_frame


def test_apply_ltx_mlx_patches_with_transformers_loaded():
    """Patching must not probe lazy transformers modules (would import torch)."""
    pytest.importorskip("ltx_pipelines_mlx")
    transformers = pytest.importorskip("transformers")
    from ltx_mlx_backend import _apply_ltx_mlx_patches

    _apply_ltx_mlx_patches(default_fps=24.0)
    # Ensure torch was not pulled in as a side effect of patching.
    assert "torch" not in sys.modules
    assert transformers is not None
