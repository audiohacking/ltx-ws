"""Assert ltx-ws generation paths never call system ffmpeg/ffprobe."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


def test_system_ffmpeg_discovery_disabled():
    pytest.importorskip("ltx_core_mlx")
    from ltx_core_mlx.utils import ffmpeg as ffmpeg_mod
    from ltx_mlx_backend import _FFMPEG_DISABLED_MSG, _apply_ltx_mlx_patches

    _apply_ltx_mlx_patches(default_fps=24.0)

    with pytest.raises(RuntimeError, match="PyAV-only"):
        ffmpeg_mod.find_ffmpeg()
    with pytest.raises(RuntimeError, match="PyAV-only"):
        ffmpeg_mod.find_ffprobe()
    assert "PyAV-only" in _FFMPEG_DISABLED_MSG


def test_stale_find_ffmpeg_importers_disabled():
    """Modules that bound find_ffmpeg at import must also be stubbed."""
    pytest.importorskip("ltx_pipelines_mlx")
    from ltx_mlx_backend import _apply_ltx_mlx_patches

    import ltx_core_mlx.utils.audio as audio_mod
    import ltx_core_mlx.utils.image as image_mod
    import ltx_core_mlx.utils.video as video_mod
    import ltx_pipelines_mlx.utils.media_io as media_mod

    _apply_ltx_mlx_patches(default_fps=24.0)

    for mod in (image_mod, video_mod, audio_mod, media_mod):
        with pytest.raises(RuntimeError, match="PyAV-only"):
            mod.find_ffmpeg()


def test_guide_module_uses_pyav_not_ffmpeg():
    """Face-swap guide encode must not keep upstream ffmpeg bindings."""
    pytest.importorskip("ltx_core_mlx")
    import ltx_ltxv_add_guide as guide
    import ltx_media
    from ltx_mlx_backend import _VIDEO_IO_STALE_IMPORTERS, _apply_ltx_mlx_patches

    _apply_ltx_mlx_patches(default_fps=24.0)
    assert guide.probe_video_info is ltx_media.probe_video_info
    assert guide.load_video_frames_normalized is ltx_media.load_video_frames_normalized
    assert "ltx_ltxv_add_guide" in _VIDEO_IO_STALE_IMPORTERS


def test_no_direct_ffmpeg_binary_argv_in_first_party():
    """First-party Python must not spawn ffmpeg/ffprobe by binary name."""
    root = Path(__file__).resolve().parents[1]
    banned = frozenset({"ffmpeg", "ffprobe"})
    offenders: list[str] = []

    for path in sorted(root.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                elts = None
                if isinstance(arg, (ast.List, ast.Tuple)):
                    elts = arg.elts
                if not elts:
                    continue
                for elt in elts:
                    if isinstance(elt, ast.Constant) and elt.value in banned:
                        offenders.append(f"{path.name}:{elt.lineno}:{elt.value!r}")

    assert not offenders, f"Direct ffmpeg/ffprobe argv found: {offenders}"
