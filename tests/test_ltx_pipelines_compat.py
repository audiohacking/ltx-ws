"""Compat patches for ltx-pipelines-mlx API drift."""

from __future__ import annotations

import inspect
import types

import pytest


def test_patch_combined_image_conditionings_default_frame_rate(monkeypatch):
    def original(images, *, enc_h, enc_w, spatial_dims, video_encoder, frame_rate):
        return frame_rate

    orch = types.SimpleNamespace(
        combined_image_conditionings=original,
        _ltx_ws_frame_rate_patched=False,
    )
    fake_pkg = types.SimpleNamespace(utils=types.SimpleNamespace(_orchestration=orch))
    monkeypatch.setitem(__import__("sys").modules, "ltx_pipelines_mlx", fake_pkg)
    monkeypatch.setitem(__import__("sys").modules, "ltx_pipelines_mlx.utils", fake_pkg.utils)
    monkeypatch.setitem(__import__("sys").modules, "ltx_pipelines_mlx.utils._orchestration", orch)

    from ltx_mlx_backend import _patch_ltx_pipelines_compat

    _patch_ltx_pipelines_compat(default_fps=24.0)
    patched = orch.combined_image_conditionings
    assert patched is not original
    assert inspect.signature(patched).parameters["frame_rate"].default == 24.0
    assert patched(
        [],
        enc_h=1,
        enc_w=1,
        spatial_dims=(1, 1, 1),
        video_encoder=object(),
    ) == 24.0
