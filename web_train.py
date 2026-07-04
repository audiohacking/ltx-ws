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


class TrainJobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


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
    _load_train_jobs_from_disk(state)


def _record_from_payload(job_id: str, payload: dict[str, Any]) -> TrainJobRecord:
    validation = payload.get("validation_clips")
    if validation is None:
        validation = []
    return TrainJobRecord(
        id=job_id,
        name=str(payload.get("name") or job_id),
        preset=str(payload.get("preset") or "t2v"),
        status=str(payload.get("status") or "queued"),
        created_at=str(payload.get("created_at") or ""),
        phase=str(payload.get("phase") or payload.get("status") or "queued"),
        step=int(payload.get("step") or 0),
        total_steps=int(payload.get("total_steps") or 0),
        error=payload.get("error"),
        artifact_url=payload.get("artifact_url"),
        artifact_name=(
            Path(str(payload["artifact_lora"])).name
            if payload.get("artifact_lora") and not payload.get("artifact_name")
            else payload.get("artifact_name")
        ),
        registered_lora_id=payload.get("registered_lora_id"),
        validation_clips=list(validation) if isinstance(validation, list) else [],
    )


def _load_train_jobs_from_disk(state: AppState) -> None:
    from ltx_train_backend import (
        discover_train_job_ids,
        job_api_payload,
        reconcile_interrupted_jobs,
    )

    reconcile_interrupted_jobs(state.output_dir)
    for job_id in discover_train_job_ids(state.output_dir):
        payload = job_api_payload(state.output_dir, job_id)
        if not payload:
            continue
        state.train_jobs[job_id] = _record_from_payload(job_id, payload)


def is_training_active(state: AppState) -> bool:
    _init_train_state(state)
    return state._active_train_job_id is not None


async def emit_train(state: AppState, job_id: str, event: dict[str, Any]) -> None:
    _init_train_state(state)
    q = state.train_event_queues.get(job_id)
    if q:
        await q.put(event)


def _job_for_api(state: AppState, job: TrainJobRecord | None = None, *, job_id: str | None = None) -> dict[str, Any]:
    from ltx_train_backend import job_api_payload

    jid = job_id or (job.id if job else "")
    payload = job_api_payload(state.output_dir, jid)
    if payload:
        return payload
    if job:
        return asdict(job)
    return {}


def _sync_job_record(state: AppState, job_id: str) -> None:
    from ltx_train_backend import job_api_payload

    payload = job_api_payload(state.output_dir, job_id)
    if not payload:
        return
    state.train_jobs[job_id] = _record_from_payload(job_id, payload)


async def _execute_train_job(state: AppState, job_id: str) -> None:
    from ltx_train_backend import (
        TrainingCancelledError,
        load_manifest,
        parse_train_request,
        run_train_job,
    )

    _init_train_state(state)
    job = state.train_jobs.get(job_id)
    if not job:
        _sync_job_record(state, job_id)
        job = state.train_jobs.get(job_id)
    if not job:
        return

    manifest = load_manifest(state.output_dir, job_id) or {}
    req = parse_train_request(manifest)

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
            _sync_job_record(state, job_id)

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
            _sync_job_record(state, job_id)
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
            _sync_job_record(state, job_id)
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
                from ltx_train_backend import load_status, save_status

                status = load_status(state.output_dir, job_id) or {}
                status["phase"] = "failed"
                status["status"] = "failed"
                status["error"] = str(exc)
                save_status(state.output_dir, job_id, status)
                _sync_job_record(state, job_id)
                await emit_train(state, job_id, {"type": "error", "message": str(exc)})
        finally:
            await emit_train(
                state,
                job_id,
                {"type": "job_complete", "job_id": job_id},
            )


def ensure_train_worker(state: AppState) -> None:
    _init_train_state(state)
    if state._train_worker_started:
        return
    asyncio.create_task(_train_worker_loop(state))
    state._train_worker_started = True


def _save_uploads(uploads: list[Any], dest_dir: Path) -> int:
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for upload in uploads:
        name = Path(upload.filename or f"upload_{saved}").name
        dest = dest_dir / name
        with dest.open("wb") as fh:
            shutil.copyfileobj(upload.file, fh)
        saved += 1
    return saved


def register_train_routes(app: Any, state: AppState) -> None:
    from fastapi import File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, StreamingResponse
    from ltx_train_backend import (
        VIDEO_EXTENSIONS,
        job_api_payload,
        job_root,
        load_manifest,
        load_status,
        parse_train_request,
        register_trained_lora,
        save_manifest,
        save_status,
        trainer_health,
        training_job_paths,
        find_latest_training_checkpoint,
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
        _load_train_jobs_from_disk(state)
        jobs = sorted(
            state.train_jobs.values(),
            key=lambda j: j.created_at,
            reverse=True,
        )
        return {"jobs": [_job_for_api(state, j) for j in jobs]}

    @app.get("/api/train/jobs/{job_id}")
    async def get_train_job(job_id: str):
        payload = job_api_payload(state.output_dir, job_id)
        if not payload:
            raise HTTPException(404, "Job not found")
        if job_id in state.train_jobs:
            state.train_jobs[job_id] = _record_from_payload(job_id, payload)
        return payload

    @app.post("/api/train/jobs")
    async def create_train_job(
        manifest: str = Form(...),
        videos: list[UploadFile] = File(...),
        references: list[UploadFile] | None = File(None),
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
            raise HTTPException(400, "At least one target video file is required")

        has_video = any(Path(u.filename or "").suffix.lower() in VIDEO_EXTENSIONS for u in videos)
        if not has_video:
            raise HTTPException(400, "At least one target video file is required")

        req = parse_train_request(body)
        ref_uploads = references or []
        if req.preset == "v2v":
            if not ref_uploads:
                raise HTTPException(400, "IC-LoRA requires reference videos")
            ref_video = any(Path(u.filename or "").suffix.lower() in VIDEO_EXTENSIONS for u in ref_uploads)
            if not ref_video:
                raise HTTPException(400, "IC-LoRA requires at least one reference video file")

        job_id = f"train_{uuid.uuid4().hex[:10]}"
        paths = training_job_paths(state.output_dir, job_id)
        paths.ensure_dirs()

        saved = _save_uploads(videos, paths.raw)
        if ref_uploads:
            _save_uploads(ref_uploads, paths.references)

        created_at = datetime.now().isoformat()
        body["created_at"] = created_at
        save_manifest(state.output_dir, job_id, body)

        job = TrainJobRecord(
            id=job_id,
            name=req.name,
            preset=req.preset,
            status=TrainJobStatus.QUEUED.value,
            created_at=created_at,
            total_steps=int(req.train.steps),
        )
        state.train_jobs[job_id] = job
        save_status(
            state.output_dir,
            job_id,
            {
                "job_id": job_id,
                "name": req.name,
                "preset": req.preset,
                "phase": "queued",
                "status": "queued",
                "created_at": created_at,
                "step": 0,
                "total_steps": int(req.train.steps),
                "job_dir": str(paths.root),
                "validation_clips": [],
            },
        )
        state.train_event_queues[job_id] = asyncio.Queue()
        await state._pending_train.put(job_id)
        log.info(
            "Queued train job %s  preset=%s  targets=%d  references=%d",
            job_id,
            req.preset,
            saved,
            len(ref_uploads),
        )
        return {"job_id": job_id, "name": job.name, "preset": job.preset}

    @app.post("/api/train/jobs/{job_id}/resume")
    async def resume_train_job(job_id: str):
        ensure_train_worker(state)
        payload = job_api_payload(state.output_dir, job_id)
        if not payload:
            raise HTTPException(404, "Job not found")
        status = str(payload.get("status") or "")
        if status not in (TrainJobStatus.INTERRUPTED.value, TrainJobStatus.FAILED.value):
            raise HTTPException(400, f"Job cannot be resumed from status {status!r}")
        if is_training_active(state):
            raise HTTPException(409, "A training job is already running")
        if state.is_generation_active():
            raise HTTPException(409, "Cannot start training while generation is active")
        manifest = load_manifest(state.output_dir, job_id) or {}
        ckpt_path, ckpt_step = find_latest_training_checkpoint(training_job_paths(state.output_dir, job_id))
        manifest["resume_from_checkpoint"] = True
        save_manifest(state.output_dir, job_id, manifest)

        state._cancelled_train_jobs.discard(job_id)
        save_status(
            state.output_dir,
            job_id,
            {
                **(load_status(state.output_dir, job_id) or {}),
                "phase": "queued",
                "status": "queued",
                "error": None,
            },
        )
        _sync_job_record(state, job_id)
        state.train_event_queues[job_id] = asyncio.Queue()
        await state._pending_train.put(job_id)
        log.info(
            "Resumed train job %s (checkpoint_step=%s path=%s)",
            job_id,
            ckpt_step if ckpt_path else None,
            ckpt_path,
        )
        return {
            "ok": True,
            "job_id": job_id,
            "status": "queued",
            "resume_from_checkpoint": bool(ckpt_path and ckpt_step > 0),
            "latest_checkpoint_step": ckpt_step if ckpt_path else None,
        }

    @app.post("/api/train/jobs/{job_id}/cancel")
    async def cancel_train_job(job_id: str):
        job = state.train_jobs.get(job_id)
        if not job:
            payload = job_api_payload(state.output_dir, job_id)
            if not payload:
                raise HTTPException(404, "Job not found")
            job = _record_from_payload(job_id, payload)
            state.train_jobs[job_id] = job
        if job.status in (
            TrainJobStatus.DONE.value,
            TrainJobStatus.FAILED.value,
            TrainJobStatus.CANCELLED.value,
        ):
            return {"ok": True, "status": job.status}
        state._cancelled_train_jobs.add(job_id)
        job.status = TrainJobStatus.CANCELLED.value
        status = load_status(state.output_dir, job_id) or {}
        status["phase"] = "cancelled"
        status["status"] = "cancelled"
        save_status(state.output_dir, job_id, status)
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

        payload = job_api_payload(state.output_dir, job_id)
        if not payload:
            raise HTTPException(404, "Job not found")
        job = state.train_jobs.get(job_id) or _record_from_payload(job_id, payload)

        artifact = None
        artifact_name = job.artifact_name or payload.get("artifact_name")
        if artifact_name:
            artifact = job_root(state.output_dir, job_id) / "outputs" / artifact_name
        if artifact is None and payload.get("artifact_lora"):
            artifact = Path(str(payload["artifact_lora"]))
        if artifact is None or not artifact.is_file():
            raise HTTPException(404, "Trained LoRA artifact not found")

        body = body or {}
        label = str(body.get("label") or job.name or "Trained LoRA").strip()
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

        manifest = load_manifest(state.output_dir, job_id) or {}
        manifest["registered_lora_id"] = lid
        save_manifest(state.output_dir, job_id, manifest)
        job.registered_lora_id = lid
        state.train_jobs[job_id] = job

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
        payload = job_api_payload(state.output_dir, job_id)
        if not payload:
            raise HTTPException(404, "Job not found")
        if job_id not in state.train_event_queues:
            state.train_event_queues[job_id] = asyncio.Queue()

        async def stream() -> AsyncIterator[str]:
            q = state.train_event_queues[job_id]
            snap = job_api_payload(state.output_dir, job_id) or {}
            terminal = snap.get("status") in (
                TrainJobStatus.DONE.value,
                TrainJobStatus.FAILED.value,
                TrainJobStatus.CANCELLED.value,
                TrainJobStatus.INTERRUPTED.value,
            )
            if terminal and state._active_train_job_id != job_id:
                yield f"data: {json.dumps({'type': 'snapshot', 'job': snap})}\n\n"
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
