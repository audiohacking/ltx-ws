"""
Runtime path resolution for dev installs and PyInstaller-frozen macOS apps.

Frozen layout:
  sys._MEIPASS/     — bundled read-only code + web/dist
  ~/Library/Application Support/LTX-WS/  — models, loras, outputs, logs (writable)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "LTX-WS"
DATA_DIR_ENV = "LTX_WS_DATA_DIR"


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_root() -> Path:
    """Read-only bundle root (PyInstaller _MEIPASS) or repo root in dev."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS"))
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    """Project / package root (code location)."""
    return bundle_root()


def user_data_root() -> Path:
    """Writable per-user data directory."""
    override = (os.environ.get(DATA_DIR_ENV) or "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if is_frozen():
        return Path.home() / "Library" / "Application Support" / APP_NAME
    return repo_root()


def models_dir() -> Path:
    env = (os.environ.get("VIDEOFENTANYL_MODELS") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return user_data_root() / "models"


def loras_dir() -> Path:
    env = (os.environ.get("VIDEOFENTANYL_LORA_DIR") or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return user_data_root() / "loras"


def web_outputs_dir() -> Path:
    return user_data_root() / "web_outputs"


def web_uploads_dir() -> Path:
    return user_data_root() / "web_uploads"


def web_dist_dir() -> Path:
    return bundle_root() / "web" / "dist"


def logs_dir() -> Path:
    return user_data_root() / "logs"


def configure_frozen_environment() -> None:
    """Set default env paths and create writable dirs when running as a frozen app."""
    if not is_frozen():
        return
    root = user_data_root()
    for name in ("models", "loras", "web_outputs", "web_uploads", "logs"):
        (root / name).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("VIDEOFENTANYL_MODELS", str(root / "models"))
    os.environ.setdefault("VIDEOFENTANYL_LORA_DIR", str(root / "loras"))
    os.environ.setdefault("LTX_WS_DATA_DIR", str(root))
