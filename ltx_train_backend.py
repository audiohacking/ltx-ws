# SPDX-License-Identifier: Apache-2.0
"""MLX LoRA training adapter for ltx-trainer-mlx (optional dependency).

Storage policy (no ``/tmp``):
- Per-job artifacts live under ``<web_outputs>/train/<job_id>/`` (raw uploads, clips,
  preprocessed latents, checkpoints, validation MP4s).
- Base MLX weights resolve via :func:`resolve_mlx_weights_directory` — explicit local
  path, ``$VIDEOFENTANYL_MODELS`` / ``<repo>/models/``, or an existing Hugging Face
  hub cache snapshot before downloading.
- Finished LoRAs copied for inference via :func:`register_trained_lora` into
  ``$VIDEOFENTANYL_LORA_DIR`` or ``<repo>/loras/``.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from ltx_mlx_backend import (
    LTX2_MLX_GIT_TAG,
    _local_lora_cache_dir,
    _nearest_valid_frames,
    resolve_mlx_weights_directory,
)

log = logging.getLogger("ltx_train")

REPO_ROOT = Path(__file__).resolve().parent
TRAIN_CONFIGS_DIR = REPO_ROOT / "train_configs"
DEFAULT_GEMMA = "mlx-community/gemma-3-12b-it-4bit"

TRAINER_INSTALL_HINT = (
    f'uv pip install "ltx-trainer-mlx @ git+https://github.com/dgrauet/ltx-2-mlx.git@{LTX2_MLX_GIT_TAG}'
    f'#subdirectory=packages/ltx-trainer"'
)


class TrainingCancelledError(Exception):
    """Raised when a cooperative training cancel is requested."""


@dataclass
class TrainPresetInfo:
    id: str
    label: str
    description: str
    ram_hint: str
    with_audio: bool
    low_ram_default: bool


TRAIN_PRESETS: dict[str, TrainPresetInfo] = {
    "t2v": TrainPresetInfo(
        id="t2v",
        label="Text-to-video style",
        description="Video-only LoRA on the default distilled/dev stack.",
        ram_hint="32–48 GB unified memory",
        with_audio=False,
        low_ram_default=False,
    ),
    "av": TrainPresetInfo(
        id="av",
        label="Audio + video style",
        description="Joint AV LoRA (whisper/ASMR-style); dev transformer + checkpointing.",
        ram_hint="64 GB recommended",
        with_audio=True,
        low_ram_default=True,
    ),
    "v2v": TrainPresetInfo(
        id="v2v",
        label="IC-LoRA (video-to-video)",
        description="Learn a style transfer from paired reference → target clips.",
        ram_hint="48 GB recommended",
        with_audio=False,
        low_ram_default=True,
    ),
}

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ACTIVE_JOB_PHASES = frozenset({"queued", "slicing", "preprocessing", "training", "starting"})


@dataclass
class SliceOptions:
    enabled: bool = False
    interval: float = 4.0
    res: str = "384x384"
    fps: float = 24.0
    fit: str = "crop"
    caption_template: str | None = None
    max_clips: int | None = None


@dataclass
class PreprocessOptions:
    width: int | None = 704
    height: int | None = 480
    max_frames: int = 97
    with_audio: bool = False
    frame_rate: float | None = 24.0
    reference_downscale_factor: int = 2


@dataclass
class TrainHyperparams:
    steps: int = 2000
    rank: int = 64
    learning_rate: float = 5e-4
    validation_prompts: list[str] = field(default_factory=lambda: ["a cinematic landscape at sunset"])
    validation_interval: int = 500
    checkpoint_interval: int = 500
    low_ram: bool = False
    seed: int = 42


@dataclass
class TrainJobRequest:
    preset: str
    name: str
    model_id: str
    model_dir: str | None
    slice: SliceOptions = field(default_factory=SliceOptions)
    preprocess: PreprocessOptions = field(default_factory=PreprocessOptions)
    train: TrainHyperparams = field(default_factory=TrainHyperparams)


EventCallback = Callable[[dict[str, Any]], None]
CancelCheck = Callable[[], bool]


def trainer_available() -> bool:
    try:
        import ltx_trainer_mlx  # noqa: F401

        return True
    except ImportError:
        return False


def trainer_health(*, ffmpeg_required: bool = False) -> dict[str, Any]:
    ok = trainer_available()
    ffmpeg = bool(shutil.which("ffmpeg"))
    return {
        "ok": ok and (ffmpeg or not ffmpeg_required),
        "trainer_installed": ok,
        "ffmpeg_available": ffmpeg,
        "install_hint": None if ok else TRAINER_INSTALL_HINT,
        "presets": [p.__dict__ for p in TRAIN_PRESETS.values()],
        "configs_dir": str(TRAIN_CONFIGS_DIR),
    }


def job_root(output_dir: Path, job_id: str) -> Path:
    return output_dir.resolve() / "train" / job_id


@dataclass(frozen=True)
class TrainJobPaths:
    """All mutable training artifacts for one job (under ``web_outputs``)."""

    root: Path
    raw: Path
    references: Path
    reference_clips: Path
    clips: Path
    captions: Path
    preprocessed: Path
    outputs: Path
    config: Path

    def ensure_dirs(self) -> None:
        for d in (
            self.root,
            self.raw,
            self.references,
            self.reference_clips,
            self.clips,
            self.captions,
            self.preprocessed,
            self.outputs,
        ):
            d.mkdir(parents=True, exist_ok=True)


def training_job_paths(output_dir: Path, job_id: str) -> TrainJobPaths:
    root = job_root(output_dir, job_id)
    return TrainJobPaths(
        root=root,
        raw=root / "raw",
        references=root / "references",
        reference_clips=root / "reference_clips",
        clips=root / "clips",
        captions=root / "captions",
        preprocessed=root / "preprocessed",
        outputs=root / "outputs",
        config=root / "config.yaml",
    )


def register_trained_lora(lora_path: Path, *, name: str) -> Path:
    """Copy a finished LoRA into the persistent local cache for inference presets."""
    src = lora_path.expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"LoRA weights not found: {src}")
    dest_dir = _local_lora_cache_dir()
    dest_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^\w.\-]+", "_", (name or "trained_lora").strip()).strip("._") or "trained_lora"
    dest = (dest_dir / f"{slug}.safetensors").resolve()
    shutil.copy2(src, dest)
    return dest


def status_path(output_dir: Path, job_id: str) -> Path:
    return job_root(output_dir, job_id) / "status.json"


def load_status(output_dir: Path, job_id: str) -> dict[str, Any] | None:
    path = status_path(output_dir, job_id)
    if not path.is_file():
        return None
    try:
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def save_status(output_dir: Path, job_id: str, payload: dict[str, Any]) -> None:
    root = job_root(output_dir, job_id)
    root.mkdir(parents=True, exist_ok=True)
    path = status_path(output_dir, job_id)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def manifest_path(output_dir: Path, job_id: str) -> Path:
    return job_root(output_dir, job_id) / "manifest.json"


def save_manifest(output_dir: Path, job_id: str, payload: dict[str, Any]) -> None:
    root = job_root(output_dir, job_id)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path(output_dir, job_id).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_manifest(output_dir: Path, job_id: str) -> dict[str, Any] | None:
    path = manifest_path(output_dir, job_id)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def discover_train_job_ids(output_dir: Path) -> list[str]:
    root = output_dir.resolve() / "train"
    if not root.is_dir():
        return []
    ids: list[str] = []
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "status.json").is_file():
            ids.append(child.name)
    return ids


def _phase_to_status(phase: str | None) -> str:
    key = (phase or "queued").strip().lower()
    if key in ("done",):
        return "done"
    if key in ("failed",):
        return "failed"
    if key in ("cancelled",):
        return "cancelled"
    if key in ("interrupted",):
        return "interrupted"
    if key in ACTIVE_JOB_PHASES:
        return "running"
    return "queued"


def reconcile_interrupted_jobs(output_dir: Path) -> int:
    """Mark in-flight jobs as interrupted after a process restart."""
    changed = 0
    for job_id in discover_train_job_ids(output_dir):
        status = load_status(output_dir, job_id)
        if not status:
            continue
        phase = str(status.get("phase") or "").lower()
        if phase in ACTIVE_JOB_PHASES:
            status["phase"] = "interrupted"
            status["status"] = "interrupted"
            status["error"] = status.get("error") or "Interrupted by server restart"
            save_status(output_dir, job_id, status)
            changed += 1
    return changed


def job_api_payload(output_dir: Path, job_id: str) -> dict[str, Any] | None:
    """Merge persisted status + manifest for API responses."""
    status = load_status(output_dir, job_id)
    manifest = load_manifest(output_dir, job_id)
    if not status and not manifest:
        return None
    data: dict[str, Any] = {"id": job_id}
    if status:
        data.update(status)
        data["id"] = job_id
        data["status"] = status.get("status") or _phase_to_status(status.get("phase"))
    if manifest:
        data.setdefault("name", manifest.get("name"))
        data.setdefault("preset", manifest.get("preset"))
        if manifest.get("registered_lora_id"):
            data["registered_lora_id"] = manifest["registered_lora_id"]
        data["created_at"] = manifest.get("created_at") or data.get("created_at")
    if "created_at" not in data:
        data["created_at"] = ""
    return data


def parse_train_request(body: dict[str, Any]) -> TrainJobRequest:
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
        reference_downscale_factor=int(preprocess_raw.get("reference_downscale_factor") or 2),
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


def _list_videos(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )


def _pair_reference_videos(targets: list[Path], references_dir: Path) -> list[Path]:
    paired: list[Path] = []
    for target in targets:
        direct = references_dir / target.name
        if direct.is_file():
            paired.append(direct)
            continue
        matches = sorted(
            p
            for p in references_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS and p.stem == target.stem
        )
        if not matches:
            raise ValueError(f"No reference video paired with target {target.name}")
        paired.append(matches[0])
    return paired


def _validation_reference_paths(paths: TrainJobPaths, num_prompts: int) -> list[str]:
    refs = _list_videos(paths.references)
    if not refs:
        raise ValueError("IC-LoRA requires reference videos in the job references folder")
    out: list[str] = []
    for i in range(num_prompts):
        out.append(str(refs[min(i, len(refs) - 1)].resolve()))
    return out


def encode_reference_latents(
    reference_videos: list[Path],
    *,
    preprocessed_root: Path,
    model_dir: str,
    target_height: int,
    target_width: int,
    max_frames: int,
    frame_rate: float | None,
    downscale_factor: int,
) -> None:
    """Encode paired reference clips into ``reference_latents/`` for IC-LoRA."""
    from ltx_trainer_mlx.preprocess import _encode_all_videos

    factor = max(1, int(downscale_factor))
    ref_h = max(32, int(target_height) // factor)
    ref_w = max(32, int(target_width) // factor)
    ref_h = (ref_h // 32) * 32
    ref_w = (ref_w // 32) * 32

    ref_dir = preprocessed_root / ".precomputed" / "reference_latents"
    ref_dir.mkdir(parents=True, exist_ok=True)

    _encode_all_videos(
        video_files=reference_videos,
        latents_dir=ref_dir,
        model_dir=model_dir,
        target_height=ref_h,
        target_width=ref_w,
        max_frames=max_frames,
        frame_rate=frame_rate,
    )


def _preset_yaml_path(preset: str) -> Path:
    key = (preset or "t2v").strip().lower()
    path = TRAIN_CONFIGS_DIR / f"lora_{key}.yaml"
    if not path.is_file():
        path = TRAIN_CONFIGS_DIR / "lora_t2v.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"No training preset config for {preset!r}")
    return path


def build_trainer_config(req: TrainJobRequest, *, paths: TrainJobPaths) -> Any:
    from ltx_trainer_mlx.config import LtxTrainerConfig

    raw = yaml.safe_load(_preset_yaml_path(req.preset).read_text(encoding="utf-8"))
    model_path = resolve_mlx_weights_directory(req.model_id, req.model_dir)

    model_block = dict(raw.get("model") or {})
    model_block["model_path"] = model_path
    raw["model"] = model_block
    raw["data"] = {"preprocessed_data_root": str(paths.preprocessed.resolve())}
    raw["output_dir"] = str(paths.outputs.resolve())
    raw["seed"] = int(req.train.seed)

    raw["optimization"]["steps"] = int(req.train.steps)
    raw["optimization"]["learning_rate"] = float(req.train.learning_rate)
    if req.train.low_ram:
        raw["optimization"]["enable_gradient_checkpointing"] = True

    lora = raw.get("lora") or {}
    lora["rank"] = int(req.train.rank)
    lora["alpha"] = int(req.train.rank)
    raw["lora"] = lora

    prompts = [p.strip() for p in req.train.validation_prompts if str(p).strip()]
    if not prompts:
        prompts = ["a cinematic landscape at sunset"]
    val = raw.get("validation") or {}
    val["prompts"] = prompts
    val["interval"] = int(req.train.validation_interval)
    val["skip_initial_validation"] = True
    w = int(req.preprocess.width or 704)
    h = int(req.preprocess.height or 480)
    nf = _nearest_valid_frames(int(req.preprocess.max_frames))
    val["video_dims"] = [w, h, nf]
    val["frame_rate"] = float(req.preprocess.frame_rate or 24.0)
    preset_info = TRAIN_PRESETS.get(req.preset, TRAIN_PRESETS["t2v"])
    if req.preset == "v2v":
        val["reference_videos"] = _validation_reference_paths(paths, len(prompts))
        val["reference_downscale_factor"] = max(1, int(req.preprocess.reference_downscale_factor))
        val["generate_audio"] = False
    else:
        val["generate_audio"] = bool(preset_info.with_audio)
    raw["validation"] = val

    ckpt = raw.get("checkpoints") or {}
    ckpt["interval"] = int(req.train.checkpoint_interval)
    raw["checkpoints"] = ckpt

    strat = dict(raw.get("training_strategy") or {})
    if req.preset != "v2v":
        strat["generate_audio"] = preset_info.with_audio
    raw["training_strategy"] = strat

    return LtxTrainerConfig(**raw)


@contextmanager
def _metrics_hook(on_metrics: Callable[[dict[str, float]], None] | None):
    if on_metrics is None:
        yield
        return
    from ltx_trainer_mlx import progress as progress_mod

    orig_cls = progress_mod.TrainingProgress
    orig_update = orig_cls.update_training

    def patched_update(
        self,
        *,
        loss: float,
        lr: float,
        step_time: float,
        advance: bool = True,
    ) -> None:
        orig_update(self, loss=loss, lr=lr, step_time=step_time, advance=advance)
        if advance:
            try:
                on_metrics({"loss": float(loss), "lr": float(lr), "step_time_s": float(step_time)})
            except Exception:
                pass

    progress_mod.TrainingProgress.update_training = patched_update  # type: ignore[method-assign]
    try:
        yield
    finally:
        progress_mod.TrainingProgress.update_training = orig_update  # type: ignore[method-assign]


def _check_cancel(should_cancel: CancelCheck | None) -> None:
    if should_cancel and should_cancel():
        raise TrainingCancelledError("Training cancelled")


def run_train_job(
    req: TrainJobRequest,
    *,
    output_dir: Path,
    job_id: str,
    on_event: EventCallback | None = None,
    should_cancel: CancelCheck | None = None,
) -> dict[str, Any]:
    """Execute slice → preprocess → train for one job."""
    if not trainer_available():
        raise RuntimeError(f"ltx-trainer-mlx is not installed. {TRAINER_INSTALL_HINT}")

    paths = training_job_paths(output_dir, job_id)
    paths.ensure_dirs()

    status: dict[str, Any] = {
        "job_id": job_id,
        "name": req.name,
        "preset": req.preset,
        "phase": "queued",
        "status": "running",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "step": 0,
        "total_steps": int(req.train.steps),
        "job_dir": str(paths.root),
        "model_path": None,
        "error": None,
        "validation_clips": [],
    }
    existing = load_status(output_dir, job_id)
    if existing:
        status["created_at"] = existing.get("created_at") or status["created_at"]
        status["validation_clips"] = existing.get("validation_clips") or []
    save_status(output_dir, job_id, status)

    def emit(event: dict[str, Any]) -> None:
        nonlocal status
        status.update({k: v for k, v in event.items() if k != "type"})
        status["status"] = _phase_to_status(status.get("phase"))
        save_status(output_dir, job_id, status)
        if on_event:
            on_event(event)

    videos_dir = paths.raw
    references_dir = paths.references
    captions_dir: str | None = None

    try:
        if req.preset == "v2v" and not _list_videos(paths.references):
            raise ValueError("IC-LoRA requires paired reference videos (upload to references/)")

        if req.slice.enabled:
            _check_cancel(should_cancel)
            status["phase"] = "slicing"
            emit({"type": "phase_started", "phase": "slicing", "message": "Slicing source videos…"})
            if not shutil.which("ffmpeg"):
                raise RuntimeError("ffmpeg is required for slice")
            from ltx_trainer_mlx.slice_clips import slice_videos

            sources = _list_videos(paths.raw)
            if not sources:
                raise ValueError("No video files found in upload")
            count = slice_videos(
                [str(p) for p in sources],
                str(paths.clips),
                interval=float(req.slice.interval),
                res=str(req.slice.res),
                fps=float(req.slice.fps),
                fit=str(req.slice.fit),
                caption_template=req.slice.caption_template,
                max_clips=req.slice.max_clips,
            )
            emit({"type": "phase_progress", "phase": "slicing", "message": f"Created {count} target clips"})
            videos_dir = paths.clips
            captions_dir = None

            if req.preset == "v2v":
                ref_sources = _list_videos(paths.references)
                if not ref_sources:
                    raise ValueError("IC-LoRA requires reference videos when slicing")
                ref_count = slice_videos(
                    [str(p) for p in ref_sources],
                    str(paths.reference_clips),
                    interval=float(req.slice.interval),
                    res=str(req.slice.res),
                    fps=float(req.slice.fps),
                    fit=str(req.slice.fit),
                    caption_template=req.slice.caption_template,
                    max_clips=req.slice.max_clips,
                )
                emit(
                    {
                        "type": "phase_progress",
                        "phase": "slicing",
                        "message": f"Created {ref_count} reference clips",
                    }
                )
                references_dir = paths.reference_clips
        else:
            txts = list(paths.raw.glob("*.txt"))
            if txts:
                paths.captions.mkdir(parents=True, exist_ok=True)
                for t in txts:
                    shutil.copy2(t, paths.captions / t.name)
                captions_dir = str(paths.captions)

        _check_cancel(should_cancel)
        status["phase"] = "preprocessing"
        emit({"type": "phase_started", "phase": "preprocessing", "message": "Encoding target latents…"})
        from ltx_trainer_mlx.preprocess import preprocess_dataset

        model_path = resolve_mlx_weights_directory(req.model_id, req.model_dir)
        status["model_path"] = model_path
        nf = _nearest_valid_frames(int(req.preprocess.max_frames))
        preset_info = TRAIN_PRESETS.get(req.preset, TRAIN_PRESETS["t2v"])
        with_audio = req.preprocess.with_audio or preset_info.with_audio
        preprocess_dataset(
            videos_dir=str(videos_dir),
            output_dir=str(paths.preprocessed),
            model_dir=model_path,
            gemma_model_id=DEFAULT_GEMMA,
            target_height=int(req.preprocess.height) if req.preprocess.height else None,
            target_width=int(req.preprocess.width) if req.preprocess.width else None,
            max_frames=nf,
            captions_dir=captions_dir,
            with_audio=with_audio,
            frame_rate=float(req.preprocess.frame_rate) if req.preprocess.frame_rate else None,
        )

        if req.preset == "v2v":
            _check_cancel(should_cancel)
            emit(
                {
                    "type": "phase_progress",
                    "phase": "preprocessing",
                    "message": "Encoding reference latents for IC-LoRA…",
                }
            )
            targets = _list_videos(videos_dir)
            ref_paths = _pair_reference_videos(targets, references_dir)
            if len(ref_paths) != len(targets):
                raise ValueError("Reference video count must match target clip count")
            encode_reference_latents(
                ref_paths,
                preprocessed_root=paths.preprocessed,
                model_dir=model_path,
                target_height=int(req.preprocess.height or 704),
                target_width=int(req.preprocess.width or 480),
                max_frames=nf,
                frame_rate=float(req.preprocess.frame_rate) if req.preprocess.frame_rate else None,
                downscale_factor=int(req.preprocess.reference_downscale_factor),
            )

        emit({"type": "phase_progress", "phase": "preprocessing", "message": "Preprocess complete"})

        _check_cancel(should_cancel)
        status["phase"] = "training"
        status["total_steps"] = int(req.train.steps)
        emit({"type": "phase_started", "phase": "training", "message": "Training LoRA…"})

        from ltx_trainer_mlx.trainer import LtxvTrainer

        config = build_trainer_config(req, paths=paths)
        paths.config.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")

        train_t0 = time.time()
        last_metrics: dict[str, float] = {}

        def on_metrics(m: dict[str, float]) -> None:
            last_metrics.update(m)

        def step_callback(step: int, total: int, validation_paths: list) -> None:
            _check_cancel(should_cancel)
            elapsed = max(time.time() - train_t0, 1e-6)
            eta_s = (elapsed / max(step, 1)) * max(total - step, 0)
            payload: dict[str, Any] = {
                "type": "train_step",
                "phase": "training",
                "step": int(step),
                "total_steps": int(total),
                "eta_s": round(eta_s, 1),
            }
            if last_metrics:
                payload.update(last_metrics)
            emit(payload)
            if validation_paths:
                rels = []
                for vp in validation_paths:
                    p = Path(vp)
                    try:
                        rel = p.relative_to(paths.outputs)
                    except ValueError:
                        rel = p.name
                    rels.append(
                        {
                            "step": int(step),
                            "filename": str(rel),
                            "url": f"/api/train/jobs/{job_id}/artifacts/{rel.as_posix()}",
                        }
                    )
                status.setdefault("validation_clips", []).extend(rels)
                emit({"type": "train_validation", "step": int(step), "videos": rels})

        with _metrics_hook(on_metrics):
            trainer = LtxvTrainer(config)
            saved_path, stats = trainer.train(
                disable_progress_bars=True,
                step_callback=step_callback,
            )

        lora_path = Path(saved_path)
        artifact_url = f"/api/train/jobs/{job_id}/artifacts/{lora_path.name}"
        status["phase"] = "done"
        status["status"] = "done"
        status["artifact_lora"] = str(lora_path)
        status["artifact_url"] = artifact_url
        status["stats"] = stats.model_dump() if hasattr(stats, "model_dump") else dict(stats)
        save_status(output_dir, job_id, status)
        emit(
            {
                "type": "job_done",
                "artifact_url": artifact_url,
                "artifact_name": lora_path.name,
                "stats": status["stats"],
            }
        )
        return status

    except TrainingCancelledError:
        status["phase"] = "cancelled"
        status["status"] = "cancelled"
        status["error"] = "Cancelled"
        save_status(output_dir, job_id, status)
        emit({"type": "error", "phase": status.get("phase"), "message": "Cancelled"})
        raise
    except Exception as exc:
        log.exception("Train job %s failed", job_id)
        status["phase"] = "failed"
        status["status"] = "failed"
        status["error"] = str(exc)
        save_status(output_dir, job_id, status)
        emit({"type": "error", "phase": status.get("phase"), "message": str(exc)})
        raise
