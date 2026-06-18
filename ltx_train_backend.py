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
}


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
    clips: Path
    captions: Path
    preprocessed: Path
    outputs: Path
    config: Path

    def ensure_dirs(self) -> None:
        for d in (self.root, self.raw, self.clips, self.captions, self.preprocessed, self.outputs):
            d.mkdir(parents=True, exist_ok=True)


def training_job_paths(output_dir: Path, job_id: str) -> TrainJobPaths:
    root = job_root(output_dir, job_id)
    return TrainJobPaths(
        root=root,
        raw=root / "raw",
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
    import json

    root = job_root(output_dir, job_id)
    root.mkdir(parents=True, exist_ok=True)
    path = status_path(output_dir, job_id)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    val["generate_audio"] = bool(TRAIN_PRESETS.get(req.preset, TRAIN_PRESETS["t2v"]).with_audio)
    raw["validation"] = val

    ckpt = raw.get("checkpoints") or {}
    ckpt["interval"] = int(req.train.checkpoint_interval)
    raw["checkpoints"] = ckpt

    strat = raw.get("training_strategy") or {}
    preset_info = TRAIN_PRESETS.get(req.preset, TRAIN_PRESETS["t2v"])
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
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "step": 0,
        "total_steps": int(req.train.steps),
        "job_dir": str(paths.root),
        "model_path": None,
        "error": None,
    }
    save_status(output_dir, job_id, status)

    def emit(event: dict[str, Any]) -> None:
        nonlocal status
        status.update({k: v for k, v in event.items() if k != "type"})
        save_status(output_dir, job_id, status)
        if on_event:
            on_event(event)

    videos_dir = paths.raw
    captions_dir: str | None = None

    try:
        if req.slice.enabled:
            _check_cancel(should_cancel)
            status["phase"] = "slicing"
            emit({"type": "phase_started", "phase": "slicing", "message": "Slicing source videos…"})
            if not shutil.which("ffmpeg"):
                raise RuntimeError("ffmpeg is required for slice")
            from ltx_trainer_mlx.slice_clips import slice_videos

            sources = sorted(
                p
                for p in paths.raw.iterdir()
                if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}
            )
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
            emit({"type": "phase_progress", "phase": "slicing", "message": f"Created {count} clips"})
            videos_dir = paths.clips
            captions_dir = None
        else:
            txts = list(paths.raw.glob("*.txt"))
            if txts:
                paths.captions.mkdir(parents=True, exist_ok=True)
                for t in txts:
                    shutil.copy2(t, paths.captions / t.name)
                captions_dir = str(paths.captions)

        _check_cancel(should_cancel)
        status["phase"] = "preprocessing"
        emit({"type": "phase_started", "phase": "preprocessing", "message": "Encoding latents…"})
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
        status["error"] = "Cancelled"
        save_status(output_dir, job_id, status)
        emit({"type": "error", "phase": status.get("phase"), "message": "Cancelled"})
        raise
    except Exception as exc:
        log.exception("Train job %s failed", job_id)
        status["phase"] = "failed"
        status["error"] = str(exc)
        save_status(output_dir, job_id, status)
        emit({"type": "error", "phase": status.get("phase"), "message": str(exc)})
        raise
