"""Ensure VAE decode uses PyAV instead of ffmpeg pipe."""

from __future__ import annotations

import inspect

import pytest


def test_video_decoder_uses_pyav_patch():
    pytest.importorskip("ltx_core_mlx")
    from ltx_core_mlx.model.video_vae import video_vae as vv_mod
    from ltx_mlx_backend import _patch_video_decode_pyav_only

    _patch_video_decode_pyav_only()
    assert getattr(vv_mod, "_ltx_ws_pyav_decode_patched", False)
    src = inspect.getsource(vv_mod.VideoDecoder.decode_and_stream)
    assert "stream_decoder_latent_to_mp4" in src
