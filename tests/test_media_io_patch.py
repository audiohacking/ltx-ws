"""Ensure I2V image preprocess uses PyAV instead of system ffmpeg."""

from __future__ import annotations

import inspect
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

    _patch_media_io_pyav_only()
    assert getattr(media_mod, "_ltx_ws_pyav_media_patched", False)
    src = inspect.getsource(media_mod.encode_single_frame)
    assert "find_ffmpeg" not in src
    assert "av.open" in src
