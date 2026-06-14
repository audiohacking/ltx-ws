"""
web_ui.py — WebUI API, static assets, and generation orchestration for ltx-ws.

Used by server.py (--web-ui) and optionally by web_server.py (standalone).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Optional

from starlette.requests import Request
from starlette.websockets import WebSocket

REPO_ROOT = Path(__file__).resolve().parent
log = logging.getLogger("web_ui")

KNOWN_MODELS = [
    {"id": "auto", "label": "Auto (RAM-based)", "repo": "auto"},
    {"id": "dgrauet/ltx-2.3-mlx", "label": "MLX bf16 (full quality)", "repo": "dgrauet/ltx-2.3-mlx"},
    {"id": "dgrauet/ltx-2.3-mlx-q8", "label": "MLX int8 (balanced)", "repo": "dgrauet/ltx-2.3-mlx-q8"},
    {"id": "dgrauet/ltx-2.3-mlx-q4", "label": "MLX int4 (smallest)", "repo": "dgrauet/ltx-2.3-mlx-q4"},
]

RESOLUTION_PRESETS = [
    {"id": "704x480", "width": 704, "height": 480, "label": "704 × 480"},
    {"id": "1024x576", "width": 1024, "height": 576, "label": "1024 × 576 (16:9)"},
    {"id": "576x1024", "width": 576, "height": 1024, "label": "576 × 1024 (9:16)"},
    {"id": "1280x720", "width": 1280, "height": 720, "label": "1280 × 720 (HD)"},
]

DURATION_PRESETS = [
    {"id": "2s", "seconds": 2.0, "label": "~2 seconds"},
    {"id": "4s", "seconds": 4.0, "label": "~4 seconds"},
    {"id": "5s", "seconds": 5.0, "label": "~5 seconds"},
]

GENERATION_MODES = [
    {"id": "generate", "label": "Text to video"},
    {"id": "i2v", "label": "Image to video (i2v)"},
    {"id": "a2v", "label": "Audio to video (a2v)"},
    {"id": "retake", "label": "Retake (edit region)"},
    {"id": "extend", "label": "Extend video"},
    {"id": "ic_lora", "label": "IC LoRA conditioning"},
]

CLIP_MULTIPLIER_MAX = 10
DEFAULT_OUTPUT_DIR = REPO_ROOT / "web_outputs"
DEFAULT_UPLOAD_DIR = REPO_ROOT / "web_uploads"
INDEX_FILE = "index.json"
FPS = 24
PROGRESS_KEEPALIVE_INTERVAL_S = 1.0


def _lora_catalog() -> tuple[list[dict[str, Any]], str]:
    """
    LoRA presets for the Web UI (default from LTX_WS_DEFAULT_LORA / server defaults).
    Returns (presets including a None entry, default_preset_id).
    """
    from server import (
        DEFAULT_GLOBAL_LORA_PATH,
        DEFAULT_GLOBAL_LORA_SCALE,
        ENV_DEFAULT_LORA,
        ENV_DEFAULT_LORA_SCALE,
        _default_loras_from_env,
    )

    def _label_for_spec(spec: str) -> str:
        name = spec.rsplit("/", 1)[-1] if "/" in spec else spec
        if name.endswith(".safetensors"):
            name = name[:-12]
        return name or spec

    seen: set[str] = set()
    presets: list[dict[str, Any]] = [
        {"id": "none", "label": "None (no LoRA)", "spec": "", "scale": 0.0},
    ]
    default_id = "none"

    def _add(id_: str, label: str, spec: str, scale: float, is_default: bool = False) -> None:
        nonlocal default_id
        key = f"{spec}:{scale}"
        if not spec or key in seen:
            return
        seen.add(key)
        presets.append({"id": id_, "label": label, "spec": spec, "scale": scale})
        if is_default:
            default_id = id_

    default_path = os.environ.get(ENV_DEFAULT_LORA, DEFAULT_GLOBAL_LORA_PATH).strip()
    scale_raw = os.environ.get(ENV_DEFAULT_LORA_SCALE, str(DEFAULT_GLOBAL_LORA_SCALE)).strip()
    try:
        default_scale = float(scale_raw)
    except ValueError:
        default_scale = DEFAULT_GLOBAL_LORA_SCALE

    if default_path:
        _add(
            "default",
            f"Default — {_label_for_spec(default_path)}",
            default_path,
            default_scale,
            is_default=True,
        )

    for i, (path, scale) in enumerate(_default_loras_from_env()):
        if path == default_path and scale == default_scale:
            continue
        _add(f"env_{i}", f"Env LoRA — {_label_for_spec(path)}", path, scale)

    return presets, default_id


def _ensure_lora_downloaded(spec: str) -> dict[str, Any]:
    from ltx_mlx_backend import _resolve_lora_path

    path, _ = _resolve_lora_path(spec)
    return {"ok": True, "spec": spec, "path": path}


_RUN_BODIES: dict[str, dict[str, Any]] = {}


def resolve_web_dist() -> Path:
    return REPO_ROOT / "web" / "dist"


def public_host(bind_host: str) -> str:
    if bind_host in ("0.0.0.0", "::", ""):
        return "127.0.0.1"
    return bind_host


def urls_from_request(request: Any) -> tuple[str, str]:
    """Build ws/http URLs from the incoming HTTP request (browser host)."""
    host = (
        request.headers.get("x-forwarded-host")
        or request.headers.get("host")
        or ""
    ).split(",")[0].strip()
    proto = (
        request.headers.get("x-forwarded-proto") or "http"
    ).split(",")[0].strip().lower()
    if not host:
        return "", ""
    ws_proto = "wss" if proto == "https" else "ws"
    return f"{ws_proto}://{host}/ws", f"{proto}://{host}/"


def build_server_urls(bind_host: str, port: int) -> tuple[str, str]:
    host = public_host(bind_host)
    ws_url = f"ws://{host}:{port}/ws"
    http_url = f"http://{host}:{port}/"
    return ws_url, http_url


def bind_all_http_hint(port: int) -> str:
    return f"http://<this-host>:{port}/"


def snap_frames(raw: int) -> int:
    k = max(0, round((int(raw) - 1) / 8))
    return 8 * k + 1


def duration_to_frames(seconds: float) -> int:
    return snap_frames(int(seconds * FPS))


def _clip_settings_from_body(body: dict[str, Any]) -> dict[str, Any]:
    duration_s = float(body.get("duration_seconds") or 5.0)
    clip_count = int(body.get("clip_count") or 1)
    autocontinue = bool(body.get("autocontinue", False)) or clip_count > 1
    autoconcat = bool(body.get("autoconcat", False)) or clip_count > 1
    return {
        "num_frames": body.get("num_frames") or duration_to_frames(duration_s),
        "width": body.get("width"),
        "height": body.get("height"),
        "seed": body.get("seed"),
        "num_steps": body.get("num_steps"),
        "duration_seconds": duration_s,
        "clip_count": clip_count,
        "autocontinue": autocontinue,
        "autoconcat": autoconcat,
    }


def scan_local_models() -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    models_dir = REPO_ROOT / "models"
    if not models_dir.is_dir():
        return found
    for child in sorted(models_dir.iterdir()):
        if child.is_dir():
            found.append(
                {
                    "id": str(child),
                    "label": f"Local: {child.name}",
                    "repo": str(child),
                }
            )
    return found


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class ClipRecord:
    id: str
    prompt: str
    label: str
    video_url: str
    filename: str
    chain_id: str
    clip_index: int
    mode: str
    status: str
    created_at: str
    elapsed_s: Optional[float] = None
    bytes: Optional[int] = None
    error: Optional[str] = None
    num_frames: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None
    num_steps: Optional[int] = None
    duration_seconds: Optional[float] = None
    clip_count: Optional[int] = None
    autocontinue: Optional[bool] = None
    autoconcat: Optional[bool] = None


@dataclass
class RunRecord:
    id: str
    status: str
    prompts: list[str]
    chain_id: str
    clip_ids: list[str] = field(default_factory=list)
    created_at: str = ""
    error: Optional[str] = None
    autocontinue: bool = False
    autoconcat: bool = False
    merged_url: Optional[str] = None
    merged_clip_id: Optional[str] = None


class AppState:
    def __init__(
        self,
        server_url: str,
        output_dir: Path,
        upload_dir: Path,
        preferred_model: str,
        *,
        embedded: bool = False,
        http_url: str = "",
        active_model: str = "",
        runtime_defaults: dict[str, Any] | None = None,
        server_process: Optional[subprocess.Popen] = None,
        video_server: Any = None,
    ):
        self.server_url = server_url
        self.http_url = http_url
        self.output_dir = output_dir
        self.upload_dir = upload_dir
        self.preferred_model = preferred_model
        self.active_model = active_model or preferred_model
        self.embedded = embedded
        self.runtime_defaults = runtime_defaults or {}
        self.server_process = server_process
        self.video_server = video_server
        self.runs: dict[str, RunRecord] = {}
        self.clips: dict[str, ClipRecord] = {}
        self.event_queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}
        self._pending: asyncio.Queue[str] = asyncio.Queue()
        self._worker_started = False

    def ensure_worker(self) -> None:
        """Start background generation worker (idempotent)."""
        if self._worker_started:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.load_index()
        asyncio.create_task(_worker_loop(self))
        self._worker_started = True

    def load_index(self) -> None:
        path = self.output_dir / INDEX_FILE
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for c in data.get("clips", []):
                self.clips[c["id"]] = ClipRecord(
                    **{k: v for k, v in c.items() if k in ClipRecord.__dataclass_fields__}
                )
            for r in data.get("runs", []):
                self.runs[r["id"]] = RunRecord(
                    **{k: v for k, v in r.items() if k in RunRecord.__dataclass_fields__}
                )
        except (json.JSONDecodeError, TypeError, KeyError):
            pass

    def save_index(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / INDEX_FILE
        data = {
            "clips": [asdict(c) for c in self.clips.values()],
            "runs": [asdict(r) for r in self.runs.values()],
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def clip_url(self, filename: str) -> str:
        return f"/api/videos/{filename}"

    def delete_clip_record(self, clip_id: str) -> bool:
        clip = self.clips.get(clip_id)
        if not clip:
            return False
        path = self.output_dir / clip.filename
        if path.is_file():
            try:
                path.unlink()
            except OSError as exc:
                log.warning("Could not delete clip file %s: %s", path, exc)
        del self.clips[clip_id]
        return True

    def delete_chain(self, chain_id: str) -> int:
        removed = 0
        for clip_id, clip in list(self.clips.items()):
            if clip.chain_id == chain_id:
                if self.delete_clip_record(clip_id):
                    removed += 1
        for run_id, run in list(self.runs.items()):
            if run.chain_id == chain_id:
                del self.runs[run_id]
        self.save_index()
        return removed

    async def emit(self, run_id: str, event: dict[str, Any]) -> None:
        q = self.event_queues.get(run_id)
        if q:
            await q.put(event)


def _ensure_web_deps() -> None:
    import importlib
    import subprocess

    for pkg, mod in (
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("python-multipart", "multipart"),
        ("starlette", "starlette"),
    ):
        try:
            __import__(mod)
        except ImportError:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pkg, "-q"],
                stdout=subprocess.DEVNULL,
            )
            importlib.invalidate_caches()


def _import_videofentanyl():
    from videofentanyl import (
        GenerationParams,
        Job,
        JobStatus,
        VideoSession,
        extract_last_frame,
        load_image_payload,
        load_media_payload,
        sanitize_filename,
        try_autoconcat_clips,
    )
    return (
        GenerationParams,
        Job,
        JobStatus,
        VideoSession,
        extract_last_frame,
        load_image_payload,
        load_media_payload,
        sanitize_filename,
        try_autoconcat_clips,
    )


class ProgressVideoSession:
    """VideoSession wrapper that forwards protocol events to the WebUI."""

    def __init__(self, job, mode: str, verbose: bool, on_event: Any, VideoSession):
        self._session = VideoSession(job, mode=mode, verbose=verbose)
        self._on_event = on_event
        self.job = job

    async def run(self, idle_timeout: float | None) -> bool:
        orig_json = self._session._handle_json
        orig_binary = self._session._handle_binary

        async def wrapped_json(raw: str) -> None:
            try:
                msg = json.loads(raw)
                await self._on_event({"type": "protocol", "event": msg})
            except json.JSONDecodeError:
                pass
            await orig_json(raw)

        def wrapped_binary(data: bytes) -> None:
            orig_binary(data)
            kb = sum(len(c) for c in self._session._chunks) / 1024
            asyncio.create_task(
                self._on_event(
                    {
                        "type": "download_progress",
                        "chunks": len(self._session._chunks),
                        "kb": round(kb, 1),
                    }
                )
            )

        self._session._handle_json = wrapped_json
        self._session._handle_binary = wrapped_binary
        return await self._session.run(idle_timeout)


def _set_server_override(url: str) -> None:
    import videofentanyl as vf

    vf._SERVER_OVERRIDE = url


def _run_ws_url(state: AppState) -> str:
    """Loopback WS for embedded runs — same as ``videofentanyl --server``."""
    if state.embedded and state.video_server is not None:
        return f"ws://127.0.0.1:{state.video_server.port}/ws"
    return state.server_url


async def healthcheck_ws(url: str) -> bool:
    import websockets

    try:
        async with websockets.connect(
            url,
            open_timeout=3.0,
            close_timeout=2.0,
            max_size=1024,
        ) as ws:
            await ws.send(
                json.dumps(
                    {
                        "type": "session_init_v2",
                        "preset_id": "simple_custom_prompt",
                        "curated_prompts": [],
                        "single_clip_mode": True,
                    }
                )
            )
            return True
    except Exception:
        return False


def _api_mode(mode: str) -> str:
    m = (mode or "generate").strip().lower()
    if m == "i2v":
        return "generate"
    return m


def _build_params_from_request(body: dict[str, Any]) -> Any:
    (
        GenerationParams,
        *_,
    ) = _import_videofentanyl()
    ui_mode = (body.get("mode") or "generate").strip().lower()
    mode = _api_mode(ui_mode)
    image_path = body.get("image_path")
    audio_path = body.get("audio_path")
    video_path = body.get("video_path")
    load_image_payload, load_media_payload = _import_videofentanyl()[5:7]

    image_payload = load_image_payload(image_path) if image_path else None
    audio_payload = load_media_payload(audio_path, kind="audio") if audio_path else None
    video_payload = load_media_payload(video_path, kind="video") if video_path else None

    lora_specs: list[tuple[str, float]] = []
    for item in body.get("lora_specs") or []:
        if isinstance(item, list) and len(item) == 2:
            lora_specs.append((str(item[0]), float(item[1])))

    video_conditioning_specs: list[tuple[dict, float]] = []
    for item in body.get("video_conditioning") or []:
        if isinstance(item, list) and len(item) == 2:
            payload = load_media_payload(str(item[0]).strip(), kind="video")
            video_conditioning_specs.append((payload, float(item[1])))

    duration_s = body.get("duration_seconds")
    num_frames = body.get("num_frames")
    if duration_s is not None:
        num_frames = duration_to_frames(float(duration_s))
    elif num_frames is not None:
        num_frames = snap_frames(int(num_frames))

    return GenerationParams(
        prompt=str(body.get("prompt") or "").strip(),
        preset_id="simple_custom_prompt",
        enhancement_enabled=False,
        single_clip_mode=True,
        initial_image=image_payload,
        seed=body.get("seed"),
        num_frames=num_frames,
        height=body.get("height"),
        width=body.get("width"),
        num_steps=body.get("num_steps"),
        generation_mode=mode,
        audio_input=audio_payload,
        source_video=video_payload,
        retake_start=body.get("retake_start"),
        retake_end=body.get("retake_end"),
        extend_frames=body.get("extend_frames"),
        extend_direction=body.get("extend_direction"),
        lora_specs=lora_specs,
        video_conditioning_specs=video_conditioning_specs,
    )


def _cleanup_temp_video(path: str | None) -> None:
    if not path:
        return
    try:
        p = Path(path)
        p.unlink(missing_ok=True)
        try:
            p.parent.rmdir()
        except OSError:
            pass
    except OSError:
        pass


async def _emit_protocol(on_event: Any, payload: dict[str, Any]) -> None:
    await on_event({"type": "protocol", "event": payload})


def _model_progress_payload(video_server: Any) -> dict[str, Any]:
    mp = video_server.generator.model_progress_for_ws()
    return {"model_progress": mp} if mp else {}


async def _emit_generation_progress(
    video_server: Any,
    on_event: Any,
    t_start: float,
    generation_id: str,
) -> None:
    elapsed_s = round(time.time() - t_start, 1)
    extra = _model_progress_payload(video_server)
    mp = extra.get("model_progress")
    payload: dict[str, Any] = {
        "type": "generation_progress",
        "elapsed_s": elapsed_s,
        "phase": mp.get("stage", "generating") if mp else "generating",
        "generation_id": generation_id,
        **extra,
    }
    await on_event(payload)
    await _emit_protocol(
        on_event,
        {
            "type": "generation_keepalive",
            "elapsed_s": elapsed_s,
            "phase": payload["phase"],
            "generation_id": generation_id,
            **extra,
        },
    )


async def _generation_progress_loop(
    video_server: Any,
    on_event: Any,
    t_start: float,
    generation_id: str,
    stop: asyncio.Event,
) -> None:
    while not stop.is_set():
        await _emit_generation_progress(video_server, on_event, t_start, generation_id)
        try:
            await asyncio.wait_for(stop.wait(), timeout=PROGRESS_KEEPALIVE_INTERVAL_S)
        except asyncio.TimeoutError:
            continue


async def _run_clip_inprocess(
    video_server: Any,
    job: Any,
    on_event: Any,
) -> bool:
    """Run one clip via the embedded VideoServer (no WebSocket round-trip)."""
    from videofentanyl import JobStatus

    params = job.params
    vs = video_server
    t0 = time.time()

    async def notify(**kwargs: Any) -> None:
        await _emit_protocol(on_event, kwargs)

    job.started_at = time.time()
    try:
        async with vs.scheduler.generation_slot(notify) as generation_id:
            await _emit_protocol(
                on_event,
                {
                    "type": "gpu_assigned",
                    "gpu_id": "mlx:0",
                    "session_timeout": 7200,
                    "generation_id": generation_id,
                },
            )
            await _emit_protocol(
                on_event,
                {
                    "type": "ltx2_stream_start",
                    "total_segments": 1,
                    "stream_mode": "single",
                },
            )
            await _emit_protocol(
                on_event,
                {
                    "type": "ltx2_segment_start",
                    "segment_idx": 0,
                    "total_segments": 1,
                },
            )

            progress_stop = asyncio.Event()
            progress_task = asyncio.create_task(
                _generation_progress_loop(vs, on_event, t0, generation_id, progress_stop)
            )
            try:
                video_path = await vs.generator.generate(
                    prompt=params.prompt,
                    image_data=params.initial_image,
                    audio_data=params.audio_input,
                    source_video_data=params.source_video,
                    seed=int(params.seed or 1024),
                    num_frames=params.num_frames,
                    height=params.height,
                    width=params.width,
                    mode=params.generation_mode,
                    num_steps=params.num_steps,
                    retake_start=params.retake_start,
                    retake_end=params.retake_end,
                    extend_frames=params.extend_frames,
                    extend_direction=params.extend_direction or "after",
                    lora_specs=list(params.lora_specs) if params.lora_specs else None,
                    video_conditioning_specs=(
                        list(params.video_conditioning_specs)
                        if params.video_conditioning_specs
                        else None
                    ),
                    job_id=generation_id,
                )
            finally:
                progress_stop.set()
                progress_task.cancel()
                try:
                    await progress_task
                except asyncio.CancelledError:
                    pass

            job.output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(video_path, job.output_path)
            job.file_bytes = job.output_path.stat().st_size
            job.chunk_count = 1
            job.ttff_ms = (time.time() - t0) * 1000
            job.gen_latency_ms = job.ttff_ms
            job.e2e_latency_ms = job.ttff_ms

            await on_event(
                {
                    "type": "download_progress",
                    "chunks": 1,
                    "kb": round(job.file_bytes / 1024, 1),
                }
            )
            await _emit_protocol(
                on_event,
                {
                    "type": "ltx2_segment_complete",
                    "segment_idx": 0,
                    "total_segments": 1,
                },
            )
            await _emit_protocol(on_event, {"type": "ltx2_stream_complete"})
            await _emit_protocol(
                on_event,
                {
                    "type": "latency",
                    "generation_ms": int(job.gen_latency_ms or 0),
                    "e2e_ms": int(job.e2e_latency_ms or 0),
                },
            )
            _cleanup_temp_video(video_path)
            job.finished_at = time.time()
            job.status = JobStatus.DONE
            return True
    except Exception as exc:
        job.finished_at = time.time()
        job.error = str(exc)
        job.status = JobStatus.FAILED
        await _emit_protocol(
            on_event,
            {
                "type": "error",
                "error_code": "generation_failed",
                "message": str(exc),
            },
        )
        return False


def _apply_autocontinue_frame(
    params: Any,
    i: int,
    autocontinue: bool,
    initial_image: Any,
    extract_last_frame: Any,
    prev_path: Path,
    prev_filename: str,
) -> None:
    if i == 0 and initial_image:
        params.initial_image = initial_image
        params.seed = int(time.time_ns() % (2**31 - 1)) or 1
    elif i > 0 and autocontinue:
        if prev_path.exists():
            frame = extract_last_frame(prev_path)
            if frame:
                params.initial_image = frame
                params.seed = int(time.time_ns() % (2**31 - 1)) or 1
                log.info(
                    "Web UI: autocontinue clip %d ← last frame of %s",
                    i + 1,
                    prev_filename,
                )
            else:
                log.warning(
                    "Web UI: autocontinue failed — no frame from %s",
                    prev_path,
                )
        else:
            log.warning(
                "Web UI: autocontinue failed — missing prior clip %s",
                prev_path,
            )


async def _finish_autoconcat(
    state: AppState,
    run: RunRecord,
    run_id: str,
    jobs: list[Any],
    prefix: str,
    prompts: list[str],
    gen_body: dict[str, Any],
    try_autoconcat_clips: Any,
) -> None:
    if not run.autoconcat or len(jobs) < 2:
        return
    try_autoconcat_clips(jobs, prefix, "mp4", verbose=False)
    merged_files = sorted(state.output_dir.glob(f"{prefix}_merged_*.mp4"))
    if not merged_files:
        return
    merged_path = merged_files[-1]
    merged_name = merged_path.name
    run.merged_url = state.clip_url(merged_name)
    # Fragment files were removed by autoconcat; drop their index entries.
    for clip_id in list(run.clip_ids):
        state.clips.pop(clip_id, None)
    merged_id = str(uuid.uuid4())
    max_idx = max(
        (c.clip_index for c in state.clips.values() if c.chain_id == run.chain_id),
        default=-1,
    )
    for c in state.clips.values():
        if c.chain_id == run.chain_id and c.label in ("CURRENT", "MERGED"):
            c.label = "EDIT"
    state.clips[merged_id] = ClipRecord(
        id=merged_id,
        prompt=f"{prompts[0]} (×{len(jobs)} merged)",
        label="MERGED",
        video_url=run.merged_url,
        filename=merged_name,
        chain_id=run.chain_id,
        clip_index=max_idx + 1,
        mode=str(gen_body.get("mode") or "generate"),
        status=RunStatus.DONE.value,
        created_at=datetime.now().isoformat(),
        bytes=merged_path.stat().st_size,
        **_clip_settings_from_body(gen_body),
    )
    run.merged_clip_id = merged_id
    run.clip_ids = [merged_id]
    state.save_index()
    await state.emit(
        run_id,
        {
            "type": "merged",
            "video_url": run.merged_url,
            "clip_id": merged_id,
            "filename": merged_name,
            "chain_id": run.chain_id,
        },
    )


async def _execute_run(state: AppState, run_id: str) -> None:
    if state.embedded and state.video_server is not None:
        await _execute_run_embedded(state, run_id)
        return
    await _execute_run_via_ws(state, run_id)


async def _execute_run_embedded(state: AppState, run_id: str) -> None:
    log.info("Web UI: executing run %s (in-process)", run_id)
    (
        _GenerationParams,
        Job,
        JobStatus,
        _VideoSession,
        extract_last_frame,
        _load_image,
        _load_media,
        sanitize_filename,
        try_autoconcat_clips,
    ) = _import_videofentanyl()

    run = state.runs[run_id]
    run.status = RunStatus.RUNNING.value
    await state.emit(
        run_id,
        {
            "type": "run_started",
            "run_id": run_id,
            "autoconcat": run.autoconcat,
            "clip_count": len(run.prompts),
        },
    )

    jobs: list[Job] = []
    gen_body = _RUN_BODIES.get(run_id, {})
    prompts = run.prompts
    autocontinue = run.autocontinue
    prefix = sanitize_filename(prompts[0]) or "clip"

    continue_from = gen_body.get("continue_from")
    initial_image = None
    if continue_from:
        parent = state.clips.get(continue_from)
        if parent and parent.filename:
            parent_path = state.output_dir / parent.filename
            if parent_path.exists():
                initial_image = extract_last_frame(parent_path)

    for i, prompt in enumerate(prompts):
        clip_id = run.clip_ids[i]
        clip = state.clips[clip_id]
        clip.status = RunStatus.RUNNING.value
        await state.emit(
            run_id,
            {
                "type": "clip_started",
                "clip_id": clip_id,
                "index": i,
                "total_clips": len(prompts),
            },
        )

        body = dict(gen_body)
        body["prompt"] = prompt
        params = _build_params_from_request(body)
        if i == 0 and initial_image:
            _apply_autocontinue_frame(
                params, i, True, initial_image, extract_last_frame, Path(), ""
            )
        elif i > 0 and autocontinue:
            prev_clip = state.clips[run.clip_ids[i - 1]]
            prev_path = state.output_dir / prev_clip.filename
            _apply_autocontinue_frame(
                params,
                i,
                True,
                None,
                extract_last_frame,
                prev_path,
                prev_clip.filename,
            )

        out_name = clip.filename
        job = Job(
            id=i + 1,
            params=params,
            output_path=state.output_dir / out_name,
            max_attempts=1,
        )
        jobs.append(job)

        async def on_event(event: dict[str, Any], _clip_id: str = clip_id) -> None:
            event["clip_id"] = _clip_id
            await state.emit(run_id, event)

        ok = await _run_clip_inprocess(state.video_server, job, on_event)

        if ok:
            clip.status = RunStatus.DONE.value
            clip.elapsed_s = round(job.elapsed, 2)
            clip.bytes = job.file_bytes
            clip.video_url = state.clip_url(out_name)
            for c in state.clips.values():
                if c.chain_id == run.chain_id and c.label == "CURRENT":
                    c.label = "EDIT"
            clip.label = "CURRENT"
            await state.emit(
                run_id,
                {
                    "type": "clip_done",
                    "clip_id": clip_id,
                    "video_url": clip.video_url,
                    "bytes": clip.bytes,
                    "index": i,
                    "total_clips": len(prompts),
                    "autoconcat": run.autoconcat,
                },
            )
        else:
            clip.status = RunStatus.FAILED.value
            clip.error = job.error or "Generation failed"
            run.status = RunStatus.FAILED.value
            run.error = clip.error
            state.save_index()
            await state.emit(
                run_id,
                {"type": "clip_failed", "clip_id": clip_id, "error": clip.error},
            )
            return

    await _finish_autoconcat(
        state, run, run_id, jobs, prefix, prompts, gen_body, try_autoconcat_clips
    )

    run.status = RunStatus.DONE.value
    state.save_index()
    await state.emit(
        run_id,
        {"type": "run_done", "run_id": run_id, "chain_id": run.chain_id},
    )


async def _execute_run_via_ws(state: AppState, run_id: str) -> None:
    log.info("Web UI: executing run %s", run_id)
    (
        _GenerationParams,
        Job,
        JobStatus,
        VideoSession,
        extract_last_frame,
        _load_image,
        _load_media,
        sanitize_filename,
        try_autoconcat_clips,
    ) = _import_videofentanyl()

    run = state.runs[run_id]
    run.status = RunStatus.RUNNING.value
    await state.emit(
        run_id,
        {
            "type": "run_started",
            "run_id": run_id,
            "autoconcat": run.autoconcat,
            "clip_count": len(run.prompts),
        },
    )

    _set_server_override(_run_ws_url(state))
    jobs: list[Job] = []
    gen_body = _RUN_BODIES.get(run_id, {})
    prompts = run.prompts
    autocontinue = run.autocontinue
    prefix = sanitize_filename(prompts[0]) or "clip"

    continue_from = gen_body.get("continue_from")
    initial_image = None
    if continue_from:
        parent = state.clips.get(continue_from)
        if parent and parent.filename:
            parent_path = state.output_dir / parent.filename
            if parent_path.exists():
                initial_image = extract_last_frame(parent_path)

    for i, prompt in enumerate(prompts):
        clip_id = run.clip_ids[i]
        clip = state.clips[clip_id]
        clip.status = RunStatus.RUNNING.value
        await state.emit(
            run_id,
            {
                "type": "clip_started",
                "clip_id": clip_id,
                "index": i,
                "total_clips": len(prompts),
            },
        )

        body = dict(gen_body)
        body["prompt"] = prompt
        params = _build_params_from_request(body)
        if i == 0 and initial_image:
            _apply_autocontinue_frame(
                params, i, True, initial_image, extract_last_frame, Path(), ""
            )
        elif i > 0 and autocontinue:
            prev_clip = state.clips[run.clip_ids[i - 1]]
            prev_path = state.output_dir / prev_clip.filename
            _apply_autocontinue_frame(
                params,
                i,
                True,
                None,
                extract_last_frame,
                prev_path,
                prev_clip.filename,
            )

        out_name = clip.filename
        job = Job(
            id=i + 1,
            params=params,
            output_path=state.output_dir / out_name,
            max_attempts=1,
        )
        jobs.append(job)

        async def on_event(event: dict[str, Any], _clip_id: str = clip_id) -> None:
            event["clip_id"] = _clip_id
            await state.emit(run_id, event)

        session = ProgressVideoSession(
            job, "ltx", False, on_event, VideoSession
        )
        ok = await session.run(idle_timeout=None)

        if ok:
            clip.status = RunStatus.DONE.value
            clip.elapsed_s = round(job.elapsed, 2)
            clip.bytes = job.file_bytes
            clip.video_url = state.clip_url(out_name)
            for c in state.clips.values():
                if c.chain_id == run.chain_id and c.label == "CURRENT":
                    c.label = "EDIT"
            clip.label = "CURRENT"
            await state.emit(
                run_id,
                {
                    "type": "clip_done",
                    "clip_id": clip_id,
                    "video_url": clip.video_url,
                    "bytes": clip.bytes,
                    "index": i,
                    "total_clips": len(prompts),
                    "autoconcat": run.autoconcat,
                },
            )
        else:
            clip.status = RunStatus.FAILED.value
            clip.error = job.error or "Generation failed"
            run.status = RunStatus.FAILED.value
            run.error = clip.error
            state.save_index()
            await state.emit(
                run_id,
                {"type": "clip_failed", "clip_id": clip_id, "error": clip.error},
            )
            return

    await _finish_autoconcat(
        state, run, run_id, jobs, prefix, prompts, gen_body, try_autoconcat_clips
    )

    run.status = RunStatus.DONE.value
    state.save_index()
    await state.emit(
        run_id,
        {"type": "run_done", "run_id": run_id, "chain_id": run.chain_id},
    )


async def _worker_loop(state: AppState) -> None:
    while True:
        run_id = await state._pending.get()
        try:
            await _execute_run(state, run_id)
        except Exception as exc:
            run = state.runs.get(run_id)
            if run:
                run.status = RunStatus.FAILED.value
                run.error = str(exc)
                state.save_index()
                await state.emit(run_id, {"type": "error", "message": str(exc)})
        finally:
            await state.emit(
                run_id,
                {
                    "type": "run_complete",
                    "run_id": run_id,
                    "chain_id": state.runs.get(run_id).chain_id if state.runs.get(run_id) else None,
                },
            )


def create_app(
    state: AppState,
    mount_static: bool = True,
    ws_handler: Callable[..., Any] | None = None,
) -> Any:
    _ensure_web_deps()
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state.ensure_worker()
        yield
        if state.server_process:
            state.server_process.terminate()

    app = FastAPI(title="ltx-ws WebUI", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def _defaults() -> dict[str, Any]:
        if state.runtime_defaults:
            return dict(state.runtime_defaults)
        return {
            "num_frames": duration_to_frames(5.0),
            "width": 704,
            "height": 480,
            "num_steps": 8,
            "fps": FPS,
        }

    async def _is_connected(request: Request | None = None) -> bool:
        if state.embedded:
            return True
        url = state.server_url
        if request is not None:
            ws_url, _ = urls_from_request(request)
            if ws_url:
                url = ws_url
        return await healthcheck_ws(url)

    @app.get("/api/health")
    async def api_health(request: Request):
        ws_url, http_url = urls_from_request(request)
        if ws_url:
            state.server_url = ws_url
        if http_url:
            state.http_url = http_url
        ok = await _is_connected(request)
        return {"ok": ok, "server_url": state.server_url, "web_url": state.http_url}

    @app.get("/api/config")
    async def api_config(request: Request):
        ws_url, http_url = urls_from_request(request)
        if ws_url:
            state.server_url = ws_url
        if http_url:
            state.http_url = http_url
        local = scan_local_models()
        models = KNOWN_MODELS + local
        ok = await _is_connected(request)
        lora_presets, default_lora_preset_id = _lora_catalog()
        model_note = (
            "MLX weights only (dgrauet/ltx-2.3-mlx*). "
            "Restart server.py with --model when changing model."
        )
        if state.embedded:
            model_note = (
                f"Active model: {state.active_model}. "
                "Restart server.py with --model <repo> to change weights."
            )
        return {
            "server_connected": ok,
            "embedded": state.embedded,
            "server_url": state.server_url,
            "web_url": state.http_url,
            "active_model": state.active_model,
            "preferred_model": state.preferred_model,
            "models": models,
            "resolution_presets": RESOLUTION_PRESETS,
            "duration_presets": DURATION_PRESETS,
            "generation_modes": GENERATION_MODES,
            "clip_multiplier_max": CLIP_MULTIPLIER_MAX,
            "defaults": _defaults(),
            "model_note": model_note,
            "lora_presets": lora_presets,
            "default_lora_preset_id": default_lora_preset_id,
        }

    @app.post("/api/loras/ensure")
    async def ensure_lora(body: dict[str, Any]):
        spec = str(body.get("spec") or "").strip()
        if not spec:
            raise HTTPException(400, "spec is required")
        try:
            result = _ensure_lora_downloaded(spec)
        except Exception as exc:
            log.warning("LoRA ensure failed for %s: %s", spec, exc)
            raise HTTPException(500, f"LoRA download failed: {exc}") from exc
        return result

    @app.post("/api/config/model")
    async def set_model(body: dict[str, Any]):
        model = str(body.get("model") or "auto").strip()
        state.preferred_model = model
        if state.embedded:
            return {
                "preferred_model": state.preferred_model,
                "active_model": state.active_model,
                "server_restarted": False,
                "server_connected": True,
                "note": "Restart server.py with --model to load different weights.",
            }
        spawned = False
        if state.server_process:
            state.server_process.terminate()
            state.server_process = None
        if body.get("restart_server"):
            cmd = [
                sys.executable,
                str(REPO_ROOT / "server.py"),
                "--model",
                model,
                "--host",
                "127.0.0.1",
                "--port",
                "8765",
            ]
            state.server_process = subprocess.Popen(
                cmd,
                cwd=str(REPO_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            spawned = True
            for _ in range(60):
                await asyncio.sleep(2)
                if await healthcheck_ws(state.server_url):
                    break
        return {
            "preferred_model": state.preferred_model,
            "server_restarted": spawned,
            "server_connected": await _is_connected(),
        }

    @app.get("/api/clips")
    async def list_clips(chain_id: Optional[str] = None):
        clips = list(state.clips.values())
        if chain_id:
            clips = [c for c in clips if c.chain_id == chain_id]
        clips.sort(key=lambda c: c.created_at)
        return {"clips": [asdict(c) for c in clips]}

    @app.delete("/api/clips/{clip_id}")
    async def delete_clip(clip_id: str):
        if not state.delete_clip_record(clip_id):
            raise HTTPException(404, "Clip not found")
        state.save_index()
        return {"ok": True, "deleted": clip_id}

    @app.delete("/api/chains/{chain_id}")
    async def delete_chain(chain_id: str):
        count = state.delete_chain(chain_id)
        if count == 0:
            raise HTTPException(404, "Chain not found")
        return {"ok": True, "deleted": count, "chain_id": chain_id}

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str):
        run = state.runs.get(run_id)
        if not run:
            raise HTTPException(404, "Run not found")
        return asdict(run)

    @app.post("/api/generate")
    async def generate(body: dict[str, Any]):
        prompt = str(body.get("prompt") or "").strip()
        prompts = body.get("prompts") or []
        if prompt:
            prompts = [prompt] + [p for p in prompts if p.strip()]
        prompts = [p.strip() for p in prompts if p and str(p).strip()]
        if not prompts:
            raise HTTPException(400, "prompt is required")

        ui_mode = (body.get("mode") or "generate").strip().lower()
        continue_from = body.get("continue_from")
        clip_count = int(body.get("clip_count") or 1)
        clip_count = max(1, min(CLIP_MULTIPLIER_MAX, clip_count))
        # Multi-clip runs are one self-contained autocontinue chain (README --count N).
        if clip_count > 1:
            continue_from = None
            chain_id = str(uuid.uuid4())
        else:
            chain_id = body.get("chain_id") or str(uuid.uuid4())

        if ui_mode == "i2v" and not body.get("image_path") and not continue_from:
            raise HTTPException(400, "i2v mode requires an image upload")
        if ui_mode == "a2v" and not body.get("audio_path"):
            raise HTTPException(400, "a2v mode requires an audio upload")
        api_mode = _api_mode(ui_mode)
        if api_mode in ("retake", "extend") and not body.get("video_path"):
            raise HTTPException(400, f"{ui_mode} mode requires video")
        if ui_mode == "ic_lora":
            if not body.get("lora_specs"):
                raise HTTPException(400, "ic_lora requires lora_specs")
            if not body.get("video_conditioning"):
                raise HTTPException(400, "ic_lora requires video_conditioning")

        if clip_count > 1 and len(prompts) == 1:
            prompts = [prompts[0]] * clip_count

        mode = ui_mode
        run_id = str(uuid.uuid4())
        autocontinue = bool(body.get("autocontinue", False)) or clip_count > 1
        autoconcat = bool(body.get("autoconcat", False)) or clip_count > 1
        if autoconcat:
            autocontinue = True

        body = dict(body)
        if clip_count > 1:
            body.pop("continue_from", None)
            body["chain_id"] = chain_id

        existing = [c for c in state.clips.values() if c.chain_id == chain_id]
        base_index = len(existing)

        clip_ids: list[str] = []
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        from videofentanyl import sanitize_filename as _sanitize

        for i, p in enumerate(prompts):
            clip_id = str(uuid.uuid4())
            slug = _sanitize(p) or "clip"
            filename = f"web_{slug}_{ts}_{i}.mp4"
            label = (
                "ORIGINAL"
                if base_index == 0 and i == 0 and not continue_from
                else "CURRENT"
            )
            clip = ClipRecord(
                id=clip_id,
                prompt=p,
                label=label,
                video_url="",
                filename=filename,
                chain_id=chain_id,
                clip_index=base_index + i,
                mode=mode,
                status=RunStatus.QUEUED.value,
                created_at=datetime.now().isoformat(),
                **_clip_settings_from_body(body),
            )
            state.clips[clip_id] = clip
            clip_ids.append(clip_id)

        run = RunRecord(
            id=run_id,
            status=RunStatus.QUEUED.value,
            prompts=prompts,
            chain_id=chain_id,
            clip_ids=clip_ids,
            created_at=datetime.now().isoformat(),
            autocontinue=autocontinue or (continue_from is not None),
            autoconcat=autoconcat,
        )
        state.runs[run_id] = run
        _RUN_BODIES[run_id] = body
        state.save_index()

        state.event_queues[run_id] = asyncio.Queue()
        await state._pending.put(run_id)
        log.info(
            "Web UI: queued run %s  chain=%s  clips=%d  mode=%s",
            run_id,
            chain_id,
            len(clip_ids),
            mode,
        )

        return {"run_id": run_id, "chain_id": chain_id, "clip_ids": clip_ids}

    @app.get("/api/runs/{run_id}/events")
    async def run_events(run_id: str):
        if run_id not in state.runs:
            raise HTTPException(404, "Run not found")
        if run_id not in state.event_queues:
            state.event_queues[run_id] = asyncio.Queue()

        async def stream() -> AsyncIterator[str]:
            q = state.event_queues[run_id]
            run = state.runs[run_id]
            if run.status in (RunStatus.DONE.value, RunStatus.FAILED.value):
                if run.status == RunStatus.DONE.value and run.merged_clip_id:
                    merged = state.clips.get(run.merged_clip_id)
                    if merged and merged.video_url:
                        yield f"data: {json.dumps({
                            'type': 'merged',
                            'video_url': merged.video_url,
                            'clip_id': merged.id,
                            'filename': merged.filename,
                            'chain_id': merged.chain_id,
                        })}\n\n"
                elif run.status == RunStatus.DONE.value:
                    for clip_id in run.clip_ids:
                        clip = state.clips.get(clip_id)
                        if clip and clip.status == RunStatus.DONE.value and clip.video_url:
                            yield f"data: {json.dumps({
                                'type': 'clip_done',
                                'clip_id': clip.id,
                                'video_url': clip.video_url,
                                'bytes': clip.bytes,
                            })}\n\n"
                yield f"data: {json.dumps({'type': 'run_complete', 'run_id': run_id, 'chain_id': run.chain_id})}\n\n"
                return
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=120.0)
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("type") in ("run_complete", "run_done", "error"):
                        if event.get("type") in ("run_complete", "run_done"):
                            break
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...), kind: str = "image"):
        ext = Path(file.filename or "upload.bin").suffix or ".bin"
        uid = str(uuid.uuid4())
        dest = state.upload_dir / f"{uid}{ext}"
        content = await file.read()
        dest.write_bytes(content)
        return {"path": str(dest), "filename": file.filename, "kind": kind}

    @app.get("/api/videos/{filename}")
    async def serve_video(filename: str):
        path = state.output_dir / filename
        if not path.is_file():
            raise HTTPException(404, "Video not found")
        return FileResponse(path, media_type="video/mp4", filename=filename)

    if ws_handler is not None:
        @app.websocket("/ws")
        async def websocket_inference(ws: WebSocket) -> None:
            # Must accept in this route — delegating accept() to ws_handler alone
            # yields HTTP 403 on the WebSocket upgrade (Starlette/FastAPI requirement).
            await ws.accept()
            await ws_handler(ws)

    if mount_static and resolve_web_dist().is_dir():
        app.mount("/", StaticFiles(directory=str(resolve_web_dist()), html=True), name="static")

    return app


def build_combined_application(
    ws_handler: Callable[..., Any],
    state: AppState,
) -> Any:
    """Single FastAPI app: WebSocket /ws + HTTP API + static UI (lifespan runs)."""
    return create_app(state, mount_static=True, ws_handler=ws_handler)


async def run_uvicorn(app: Any, host: str, port: int) -> None:
    _ensure_web_deps()
    import uvicorn

    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()


def run_standalone() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ltx-ws WebUI (standalone)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5299)
    parser.add_argument(
        "--server-url",
        default=os.environ.get(
            "LTX_WS_URL", build_server_urls("127.0.0.1", 8765)[0]
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--upload-dir", type=Path, default=DEFAULT_UPLOAD_DIR)
    parser.add_argument(
        "--model",
        default=os.environ.get("LTX_WS_MODEL", "auto"),
    )
    parser.add_argument("--spawn-server", action="store_true")
    args = parser.parse_args()

    server_proc = None
    if args.spawn_server:
        server_proc = subprocess.Popen(
            [
                sys.executable,
                str(REPO_ROOT / "server.py"),
                "--model",
                args.model,
                "--web-ui",
                "--host",
                "127.0.0.1",
                "--port",
                "8765",
            ],
            cwd=str(REPO_ROOT),
        )

    ws_url = args.server_url
    http_url = f"http://{public_host(args.host)}:{args.port}/"
    state = AppState(
        server_url=ws_url,
        output_dir=args.output_dir.resolve(),
        upload_dir=args.upload_dir.resolve(),
        preferred_model=args.model,
        http_url=http_url,
        server_process=server_proc,
    )
    app = create_app(state)
    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
