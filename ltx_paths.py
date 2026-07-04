"""Scratch paths for temp media — never macOS TMPDIR (/var/folders/...)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

_configured_root: Path | None = None


def configure_scratch_root(root: Path | str | None) -> None:
    """Prefer ``output_dir/.scratch``, else ``/tmp/ltx-ws`` (see :func:`scratch_root`)."""
    global _configured_root
    if root is None:
        _configured_root = None
        return
    _configured_root = Path(root).expanduser().resolve()


def scratch_root() -> Path:
    if _configured_root is not None:
        return _configured_root
    env = os.environ.get("LTX_SCRATCH_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return Path("/tmp/ltx-ws")


def ensure_scratch_root() -> Path:
    root = scratch_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def mk_scratch_dir(prefix: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix, dir=str(ensure_scratch_root())))


def mk_scratch_file(prefix: str, suffix: str) -> tuple[int, str]:
    return tempfile.mkstemp(prefix=prefix, suffix=suffix, dir=str(ensure_scratch_root()))
