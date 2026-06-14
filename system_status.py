"""
In-app system status for frozen macOS builds (no console).

Phases: idle | downloading_model | downloading_lora | loading_mlx | loading_pipeline
        | resolving_loras | ready | error

SSE: subscribers receive JSON snapshots on change.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from typing import Any, AsyncIterator, Deque

_lock = threading.Lock()
_phase = "idle"
_message = "Starting…"
_detail = ""
_pct: float | None = None
_model: str | None = None
_pipeline: str | None = None
_error: str | None = None
_log: Deque[str] = deque(maxlen=200)
_subscribers: list[asyncio.Queue[dict[str, Any]]] = []
_loop: asyncio.AbstractEventLoop | None = None


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def snapshot() -> dict[str, Any]:
    with _lock:
        return {
            "phase": _phase,
            "message": _message,
            "detail": _detail,
            "pct": _pct,
            "model": _model,
            "pipeline": _pipeline,
            "error": _error,
            "frozen": __import__("ltx_paths").is_frozen(),
            "log_tail": list(_log),
            "updated_at": _now_iso(),
        }


def _notify() -> None:
    snap = snapshot()
    loop = _loop
    if loop is None or not loop.is_running():
        return
    for q in list(_subscribers):
        try:
            loop.call_soon_threadsafe(q.put_nowait, snap)
        except Exception:
            pass


def bind_event_loop(loop: asyncio.AbstractEventLoop) -> None:
    global _loop
    _loop = loop


def log(line: str) -> None:
    text = (line or "").strip()
    if not text:
        return
    with _lock:
        _log.append(text)
    _notify()


def set_status(
    phase: str,
    message: str,
    *,
    detail: str = "",
    pct: float | None = None,
    model: str | None = None,
    pipeline: str | None = None,
    error: str | None = None,
) -> None:
    global _phase, _message, _detail, _pct, _model, _pipeline, _error
    with _lock:
        _phase = phase
        _message = message
        _detail = detail
        if pct is not None:
            _pct = pct
        if model is not None:
            _model = model
        if pipeline is not None:
            _pipeline = pipeline
        if error is not None:
            _error = error
        if phase != "error":
            _error = None
    log(f"[{phase}] {message}" + (f" — {detail}" if detail else ""))
    _notify()


def set_download_progress(pct: float, message: str, detail: str = "") -> None:
    global _phase, _message, _detail, _pct
    with _lock:
        phase = _phase if _phase.startswith("downloading") else "downloading_model"
        pct_val = max(0.0, min(100.0, float(pct)))
        _phase = phase
        _pct = pct_val
        _message = message
        _detail = detail
    set_status(phase, message, detail=detail, pct=pct_val)


async def subscribe() -> AsyncIterator[dict[str, Any]]:
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    _subscribers.append(q)
    try:
        await q.put(snapshot())
        while True:
            yield await q.get()
    finally:
        _subscribers.remove(q)


class StatusTqdm:
    """Minimal tqdm stand-in for huggingface_hub download progress."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.total = kwargs.get("total") or (args[0] if args else None)
        self.n = 0
        self.desc = str(kwargs.get("desc") or "")

    def update(self, n: float = 1) -> None:
        self.n += int(n)
        if self.total and self.total > 0:
            pct = 100.0 * self.n / float(self.total)
            set_download_progress(pct, self.desc or "Downloading…", f"{self.n}/{self.total}")

    def close(self) -> None:
        pass

    def __enter__(self) -> StatusTqdm:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
