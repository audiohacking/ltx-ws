"""Training job API, worker queue, and SSE for the ltx-ws Web UI."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, AsyncIterator

if TYPE_CHECKING:
    from web_ui import AppState

log = logging.getLogger("web_train")

_TRAIN_BODIES: dict[str, dict[str, Any]] = {}


class TrainJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TrainJobRecord:
    id: str
    name: str
    preset: str
    status: str
    created_at: str
    phase: str = "queued"
    step: int = 0
    total_steps: int = 0
    error: str | None = None
    artifact_url: str | None = None
    artifact_name: str | None = None
    registered_lora_id: str | None = None
    validation_clips: list[dict[str, Any]] = field(default_factory=list)


def _init_train_state(state: AppState) -> None:
    if getattr(state, "_train_initialized", False):
        return
    state.train_jobs: dict[str, TrainJobRecord] = {}
    state.train_event_queues: dict[str, asyncio.Queue[dict[str, Any]]] = {}
    state._pending_train: asyncio.Queue[str] = asyncio.Queue()
    state._cancelled_train_jobs: set[str] = set()
    state._active_train_job_id: str | None = None
    state._train_worker_started = False
    state._mlx_lock = asyncio.Lock()
    state._train_initialized = True


def is_training_active(state: AppState) -> bool:
    _init_train_state(state)
    return state._active_train_job_id is not None


async def emit_train(state: AppState, job_id: str, event: dict[str, Any]) -> None:
    _init_train_state(state)
    q = state.train_event_queues.get(job_id)
    if q:
        await q.put(event)


def _job_for_api(state: AppState, job: TrainJobRecord) -> dict[str, Any]:
    data = asdict(job)
    status = None
    try:
        from ltx_train_backend import load_status

        status = load_status(state.output_dir, job.id)
    except Exception:
        pass
    if status:
        for key in (
            "phase",
            "step",
            "total_steps",
            "loss",
            "lr",
            "eta_s",
            "artifact_url",
            "artifact_lora",
            "validation_clips",
            "error",
            "stats",
        ):
            if key in status and status[key] is not None:
                data[key] = status[key]
    return data


def _parse_train_request(body: dict[str, Any]) -> Any:
    from ltx_train_backend import (
        PreprocessOptions,
        SliceOptions,
        TrainHyperparams,
        TrainJobRequest,
        TRAIN_PRESETS,
    )

    preset = str(body.get("preset") or "t2v").strip().lower()
    if preset not in TRAIN_PRESETS:
        preset = "t2v"
    preset_info = TRAIN_PRESETS[preset]

    slice_raw = body.get("slice") or {}
    preprocess_raw = body.get("preprocess") or {}
    train_raw = body.get("train") or {}

    slice_opts = SliceOptions(
        enabled=bool(slice_raw.get("enabled", False)),
        interval=float(slice_raw.get("interval", 4.0)),
        res=str(slice_raw.get("res") or "384x384"),
        fps=float(slice_raw.get("fps", 24.0)),
        fit=str(slice_raw.get("fit") or "crop"),
        caption_template=slice_raw.get("caption_template"),
        max_clips=slice_raw.get("max_clips"),
    )
    preprocess = PreprocessOptions(
        width=int(preprocess_raw.get("width") or 704),
        height=int(preprocess_raw.get("height") or 480),
        max_frames=int(preprocess_raw.get("max_frames") or 97),
        with_audio=bool(preprocess_raw.get("with_audio", preset_info.with_audio)),
        frame_rate=float(preprocess_raw.get("frame_rate") or 24.0),
    )
    prompts_raw = train_raw.get("validation_prompts")
    if isinstance(prompts_raw, str):
        prompts = [p.strip() for p in prompts_raw.split("\n") if p.strip()]
    elif isinstance(prompts_raw, list):
        prompts = [str(p).strip() for p in prompts_raw if str(p).strip()]
    else:
        prompts = ["a cinematic landscape at sunset"]

    train = TrainHyperparams(
        steps=int(train_raw.get("steps") or 2000),
        rank=int(train_raw.get("rank") or 64),
        learning_rate=float(train_raw.get("learning_rate") or 5e-4),
        validation_prompts=prompts,
        validation_interval=int(train_raw.get("validation_interval") or 500),
        checkpoint_interval=int(train_raw.get("checkpoint_interval") or 500),
        low_ram=bool(train_raw.get("low_ram", preset_info.low_ram_default)),
        seed=int(train_raw.get("seed") or 42),
    )

    return TrainJobRequest(
        preset=preset,
        name=str(body.get("name") or "My LoRA").strip() or "My LoRA",
        model_id=str(body.get("model_id") or body.get("preferred_model") or "auto"),
        model_dir=body.get("model_dir"),
        slice=slice_opts,
        preprocess=preprocess,
        train=train,
    )


async def _execute_train_job(state: AppState, job_id: str) -> None:
    from ltx_train_backend import TrainingCancelledError, run_train_job

    _init_train_state(state)
    job = state.train_jobs.get(job_id)
    if not job:
        return

    body = _TRAIN_BODIES.get(job_id, {})
    req = _parse_train_request(body)

    job.status = TrainJobStatus.RUNNING.value
    job.phase = "starting"
    job.total_steps = int(req.train.steps)
    state._active_train_job_id = job_id

    loop = asyncio.get_running_loop()

    def on_event(event: dict[str, Any]) -> None:
        etype = event.get("type")
        if etype == "train_step":
            job.step = int(event.get("step") or job.step)
            job.total_steps = int(event.get("total_steps") or job.total_steps)
        elif etype == "phase_started":
            job.phase = str(event.get("phase") or job.phase)
        elif etype == "train_validation":
            videos = event.get("videos") or []
            job.validation_clips.extend(videos)
        elif etype == "job_done":
            job.artifact_url = event.get("artifact_url")
            job.artifact_name = event.get("artifact_name")

        def _schedule() -> None:
            loop.create_task(emit_train(state, job_id, event))

        loop.call_soon_threadsafe(_schedule)

    def should_cancel() -> bool:
        return job_id in state._cancelled_train_jobs

    async with state._mlx_lock:
        if is_training_active(state) and state._active_train_job_id != job_id:
            raise RuntimeError("Another training job is active")
        if state.is_generation_active():
            raise RuntimeError("Cannot train while video generation is running")

        try:
            result = await asyncio.to_thread(
                run_train_job,
                req,
                output_dir=state.output_dir,
                job_id=job_id,
                on_event=on_event,
                should_cancel=should_cancel,
            )
            job.status = TrainJobStatus.DONE.value
            job.phase = str(result.get("phase") or "done")
            job.artifact_url = result.get("artifact_url") or job.artifact_url
            if result.get("artifact_lora"):
                job.artifact_name = Path(str(result["artifact_lora"])).name
            await emit_train(
                state,
                job_id,
                {
                    "type": "job_done",
                    "artifact_url": job.artifact_url,
                    "artifact_name": job.artifact_name,
                },
            )
        except TrainingCancelledError:
            job.status = TrainJobStatus.CANCELLED.value
            job.phase = "cancelled"
            job.error = "Cancelled"
            await emit_train(state, job_id, {"type": "error", "message": "Cancelled"})
        finally:
            state._active_train_job_id = None


async def _train_worker_loop(state: AppState) -> None:
    while True:
        job_id = await state._pending_train.get()
        try:
            await _execute_train_job(state, job_id)
        except Exception as exc:
            job = state.train_jobs.get(job_id)
            if job and job.status != TrainJobStatus.CANCELLED.value:
                job.status = TrainJobStatus.FAILED.value
                job.error = str(exc)
                await emit_train(state, job_id, {"type": "error", "message": str(exc)})
        finally:
            await emit_train(
                state,
                job_id,
                {"type": "job_complete", "job_id": job_id},
            )
            _TRAIN_BODIES.pop(job_id, None)


def ensure_train_worker(state: AppState) -> None:
    _init_train_state(state)
    if state._train_worker_started:
        return
    asyncio.create_task(_train_worker_loop(state))
    state._train_worker_started = True


def register_train_routes(app: Any, state: AppState) -> None:
    from fastapi import File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, StreamingResponse
    from ltx_train_backend import (
        job_root,
        load_status,
        register_trained_lora,
        trainer_health,
        training_job_paths,
    )

    _init_train_state(state)

    @app.get("/api/train/health")
    async def train_health():
        health = trainer_health(ffmpeg_required=False)
        health["training_active"] = is_training_active(state)
        health["generation_active"] = state.is_generation_active()
        return health

    @app.get("/api/train/presets")
    async def train_presets():
        health = trainer_health(ffmpeg_required=False)
        return {"presets": health.get("presets") or []}

    @app.get("/api/train/jobs")
    async def list_train_jobs():
        jobs = sorted(
            state.train_jobs.values(),
            key=lambda j: j.created_at,
            reverse=True,
        )
        return {"jobs": [_job_for_api(state, j) for j in jobs]}

    @app.get("/api/train/jobs/{job_id}")
    async def get_train_job(job_id: str):
        job = state.train_jobs.get(job_id)
        if not job:
            status = load_status(state.output_dir, job_id)
            if not status:
                raise HTTPException(404, "Job not found")
            return status
        return _job_for_api(state, job)

    @app.post("/api/train/jobs")
    async def create_train_job(
        manifest: str = Form(...),
        videos: list[UploadFile] = File(...),
    ):
        ensure_train_worker(state)
        if is_training_active(state):
            raise HTTPException(409, "A training job is already running")
        if state.is_generation_active():
            raise HTTPException(409, "Cannot start training while generation is active")

        try:
            body = json.loads(manifest)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"Invalid manifest JSON: {exc}") from exc

        if not videos:
            raise HTTPException(400, "At least one video file is required")

        video_exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
        has_video = any(
            Path(u.filename or "").suffix.lower() in video_exts for u in videos
        )
        if not has_video:
            raise HTTPException(400, "At least one video file is required")

        job_id = f"train_{uuid.uuid4().hex[:10]}"
        paths = training_job_paths(state.output_dir, job_id)
        paths.ensure_dirs()

        saved = 0
        for upload in videos:
            name = Path(upload.filename or f"upload_{saved}").name
            dest = paths.raw / name
            with dest.open("wb") as fh:
                shutil.copyfileobj(upload.file, fh)
            saved += 1

        req = _parse_train_request(body)
        job = TrainJobRecord(
            id=job_id,
            name=req.name,
            preset=req.preset,
            status=TrainJobStatus.QUEUED.value,
            created_at=datetime.now().isoformat(),
            total_steps=int(req.train.steps),
        )
        state.train_jobs[job_id] = job
        _TRAIN_BODIES[job_id] = body
        state.train_event_queues[job_id] = asyncio.Queue()
        await state._pending_train.put(job_id)
        log.info("Queued train job %s  preset=%s  videos=%d", job_id, req.preset, saved)
        return {"job_id": job_id, "name": job.name, "preset": job.preset}

    @app.post("/api/train/jobs/{job_id}/cancel")
    async def cancel_train_job(job_id: str):
        job = state.train_jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if job.status in (
            TrainJobStatus.DONE.value,
            TrainJobStatus.FAILED.value,
            TrainJobStatus.CANCELLED.value,
        ):
            return {"ok": True, "status": job.status}
        state._cancelled_train_jobs.add(job_id)
        job.status = TrainJobStatus.CANCELLED.value
        return {"ok": True, "status": job.status}

    @app.get("/api/train/jobs/{job_id}/artifacts/{artifact_path:path}")
    async def train_artifact(job_id: str, artifact_path: str):
        root = job_root(state.output_dir, job_id) / "outputs"
        target = (root / artifact_path).resolve()
        if not str(target).startswith(str(root.resolve())):
            raise HTTPException(400, "Invalid path")
        if not target.is_file():
            raise HTTPException(404, "Artifact not found")
        media = "video/mp4" if target.suffix.lower() == ".mp4" else "application/octet-stream"
        return FileResponse(target, media_type=media, filename=target.name)

    @app.post("/api/train/jobs/{job_id}/register-lora")
    async def register_lora(job_id: str, body: dict[str, Any] | None = None):
        from web_ui import _label_for_lora_spec, _lora_catalog, _read_custom_loras, _write_custom_loras

        job = state.train_jobs.get(job_id)
        status = load_status(state.output_dir, job_id)
        artifact = None
        if job and job.artifact_url:
            artifact_name = job.artifact_name
            if artifact_name:
                artifact = job_root(state.output_dir, job_id) / "outputs" / artifact_name
        if artifact is None and status:
            al = status.get("artifact_lora")
            if al:
                artifact = Path(str(al))
        if artifact is None or not artifact.is_file():
            raise HTTPException(404, "Trained LoRA artifact not found")

        body = body or {}
        label = str(body.get("label") or (job.name if job else "") or "Trained LoRA").strip()
        try:
            scale = float(body.get("scale", 1.0))
        except (TypeError, ValueError):
            raise HTTPException(400, "scale must be a number")

        dest = register_trained_lora(artifact, name=label)
        spec = str(dest)
        lid = f"custom_{uuid.uuid4().hex[:8]}"
        entries = [
            {
                "id": e["id"],
                "label": e["label"],
                "spec": e["spec"],
                "scale": e["scale"],
            }
            for e in _read_custom_loras(state.output_dir)
        ]
        entries.append({"id": lid, "label": label or _label_for_lora_spec(spec), "spec": spec, "scale": scale})
        _write_custom_loras(state.output_dir, entries)
        if job:
            job.registered_lora_id = lid
        lora_presets, default_lora_preset_id = _lora_catalog(state.output_dir)
        preferred = state.preferred_lora_preset_ids()
        if lid not in preferred:
            preferred.append(lid)
            state.persist_preferred_loras(preferred)
        return {
            "ok": True,
            "id": lid,
            "spec": spec,
            "label": label,
            "lora_presets": lora_presets,
            "default_lora_preset_id": default_lora_preset_id,
            "preferred_lora_preset_ids": state.preferred_lora_preset_ids(),
        }

    @app.get("/api/train/jobs/{job_id}/events")
    async def train_events(job_id: str):
        if job_id not in state.train_jobs:
            status = load_status(state.output_dir, job_id)
            if not status:
                raise HTTPException(404, "Job not found")
        if job_id not in state.train_event_queues:
            state.train_event_queues[job_id] = asyncio.Queue()

        async def stream() -> AsyncIterator[str]:
            q = state.train_event_queues[job_id]
            job = state.train_jobs.get(job_id)
            if job and job.status in (
                TrainJobStatus.DONE.value,
                TrainJobStatus.FAILED.value,
                TrainJobStatus.CANCELLED.value,
            ):
                snap = _job_for_api(state, job) if job else load_status(state.output_dir, job_id) or {}
                yield f"data: {json.dumps({'type': 'snapshot', 'job': snap})}\n\n"
                yield f"data: {json.dumps({'type': 'job_complete', 'job_id': job_id})}\n\n"
                return
            status = load_status(state.output_dir, job_id)
            if status and status.get("phase") in ("done", "failed", "cancelled"):
                yield f"data: {json.dumps({'type': 'snapshot', 'job': status})}\n\n"
                yield f"data: {json.dumps({'type': 'job_complete', 'job_id': job_id})}\n\n"
                return
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=120.0)
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("type") in ("job_complete", "error", "job_done"):
                        if event.get("type") in ("job_complete", "job_done"):
                            break
                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")
