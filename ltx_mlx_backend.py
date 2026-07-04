# SPDX-License-Identifier: Apache-2.0
"""
Local LTX-2.3 generation using ``ltx-2-mlx`` (MLX on Apple Silicon).

See: https://github.com/dgrauet/ltx-2-mlx
"""

from __future__ import annotations

import asyncio
import base64
import concurrent.futures
import functools
import inspect
import logging
import mimetypes
import os
import random
import re
import shutil
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname, urlopen

log = logging.getLogger("fvserver")

LTX2_SPATIAL_ALIGN = 32
LTX2_MLX_GIT_TAG = "v0.14.12"

CHAIN_METHOD_AUTOCONTINUE = "autocontinue"
CHAIN_METHOD_NATIVE_EXTEND = "native_extend"
# ltx-2-mlx extend/retake: RetakePipeline + dev transformer + CFG (see docs/PIPELINES.md).
RETAKE_EXTEND_DEFAULT_CFG = 3.0
RETAKE_EXTEND_DEFAULT_STG = 0.0
VALID_CHAIN_METHODS = frozenset({CHAIN_METHOD_AUTOCONTINUE, CHAIN_METHOD_NATIVE_EXTEND})

PIPE_PROFILE_DISTILLED = "distilled"
PIPE_PROFILE_TWO_STAGE = "two_stage"
PIPE_PROFILE_HQ = "hq"
PIPE_PROFILE_ONE_STAGE = "one_stage"
VALID_PIPELINE_PROFILES = frozenset(
    {PIPE_PROFILE_DISTILLED, PIPE_PROFILE_TWO_STAGE, PIPE_PROFILE_HQ, PIPE_PROFILE_ONE_STAGE}
)


def ltx2_mlx_install_hint() -> str:
    return (
        "  uv pip install \\\n"
        f'    "ltx-core-mlx @ git+https://github.com/dgrauet/ltx-2-mlx.git@{LTX2_MLX_GIT_TAG}'
        '#subdirectory=packages/ltx-core-mlx" \\\n'
        f'    "ltx-pipelines-mlx @ git+https://github.com/dgrauet/ltx-2-mlx.git@{LTX2_MLX_GIT_TAG}'
        '#subdirectory=packages/ltx-pipelines-mlx"'
    )

# Hugging Face repo id: ``org/name`` (used with huggingface_hub.snapshot_download,
# same file set as ``huggingface-cli download org/name``).
_HF_REPO_ID_RE = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._-]*$"
)
REPO_ROOT = Path(__file__).resolve().parent
VIDEOFENTANYL_MODELS_ENV = "VIDEOFENTANYL_MODELS"
VIDEOFENTANYL_LORA_DIR_ENV = "VIDEOFENTANYL_LORA_DIR"
MAX_REMOTE_INPUT_BYTES = 512 * 1024 * 1024  # 512 MiB safety ceiling for remote audio/video


@dataclass
class GenerationRequest:
    prompt: str
    image_data: dict | str | None = None
    audio_data: dict | str | None = None
    source_video_data: dict | str | None = None
    seed: int = -1
    num_frames: int | None = None
    height: int | None = None
    width: int | None = None
    negative_prompt: str = ""
    mode: str = "generate"  # generate|a2v|retake|extend
    num_steps: int | None = None
    retake_start: int | None = None
    retake_end: int | None = None
    extend_frames: int | None = None
    extend_direction: str = "after"
    lora_specs: list[tuple[str, float]] | None = None
    video_conditioning_specs: list[tuple[dict | str, float]] | None = None
    job_id: str | None = None
    a2v_visual_i2v_continue: bool = False
    # Optional ltx-2-mlx advanced controls (see https://github.com/dgrauet/ltx-2-mlx#features)
    end_image_data: dict | str | None = None
    enhance_prompt: bool = False
    pipeline_profile: str = PIPE_PROFILE_DISTILLED
    cfg_scale: float | None = None
    stg_scale: float | None = None
    stage2_steps: int | None = None
    no_regen_audio: bool = False
    reference_strength: float | None = None


def looks_like_hf_repo_id(model: str) -> bool:
    """True if ``model`` looks like ``author/repo`` and is not an existing directory path."""
    s = (model or "").strip()
    if not s or _HF_REPO_ID_RE.match(s) is None:
        return False
    p = Path(s).expanduser()
    if p.is_dir():
        return False
    return True


def _snapshot_download_weights(snapshot_download: Any, repo_id: str, dest: Path) -> str:
    """Call ``snapshot_download`` with kwargs compatible across huggingface_hub versions."""
    import inspect

    kw: dict[str, Any] = {"repo_id": repo_id, "local_dir": str(dest)}
    sig = inspect.signature(snapshot_download)
    if "resume_download" in sig.parameters:
        kw["resume_download"] = True
    if "local_dir_use_symlinks" in sig.parameters:
        kw["local_dir_use_symlinks"] = False
    out = snapshot_download(**kw)
    return str(Path(out).resolve())


def _model_snapshot_present(dest: Path) -> bool:
    """
    Heuristic to detect an already materialized HF snapshot in ``dest``.
    """
    if not dest.is_dir():
        return False
    try:
        has_config = (dest / "config.json").is_file() or (dest / "embedded_config.json").is_file()
        has_weights = any(dest.glob("*.safetensors"))
    except OSError:
        return False
    return bool(has_config and has_weights)


def hf_local_weights_directory(repo_id: str, explicit_model_dir: str | None) -> Path:
    """
    Directory where we store a full ``snapshot_download`` for ``repo_id``.

    If ``explicit_model_dir`` is set, that path is used. Otherwise:
    ``$VIDEOFENTANYL_MODELS/<org>__<name>/`` when the env var is set, else
    ``<repo_root>/models/<org>__<name>/``.
    """
    rid = repo_id.strip()
    if explicit_model_dir:
        return Path(explicit_model_dir).expanduser().resolve()
    env = os.environ.get(VIDEOFENTANYL_MODELS_ENV, "").strip()
    root = Path(env).expanduser().resolve() if env else (REPO_ROOT / "models")
    safe = rid.replace("/", "__")
    return (root / safe).resolve()


def _looks_like_models_dir_leaf(name: str) -> bool:
    """True if ``name`` is a single path segment (safe to join under ``models/``)."""
    s = (name or "").strip()
    if not s or s in (".", "..") or s.startswith(".."):
        return False
    if "/" in s or "\\" in s:
        return False
    return Path(s).name == s


def _path_candidates_for_user_string(user_path: str) -> list[Path]:
    """For a filesystem path string: absolutes resolve once; relatives try cwd then repo root.

    This fixes ``python /path/to/server.py`` started from ``$HOME`` where
    ``./models/foo`` must resolve next to the checkout, not under ``$HOME``.
    """
    raw = (user_path or "").strip()
    if not raw:
        return []
    p = Path(raw).expanduser()
    if p.is_absolute():
        return [p.resolve()]
    return [(Path.cwd() / p).resolve(), (REPO_ROOT / p).resolve()]


def _first_existing_dir(user_path: str) -> Path | None:
    for c in _path_candidates_for_user_string(user_path):
        if c.is_dir():
            return c
    return None


def _resolve_non_hf_disk_path(model: str, explicit_model_dir: str | None) -> str | None:
    """
    Resolve to an existing weights directory without calling the Hub.

    Tries: ``--model`` as a directory path (cwd, then repo root for relatives),
    then ``--model-dir`` the same way, then ``models/<model>/`` under cwd and
    under repo root for a shorthand leaf (e.g. ``ltx-2.3-mlx``).
    """
    raw = (model or "").strip()
    if not raw:
        return None

    hit = _first_existing_dir(raw)
    if hit is not None:
        return str(hit)

    md = (explicit_model_dir or "").strip()
    if md:
        hit = _first_existing_dir(md)
        if hit is not None:
            return str(hit)

    if _looks_like_models_dir_leaf(raw):
        leaf = Path(raw).name
        for base in (Path.cwd(), REPO_ROOT):
            candidate = (base / "models" / leaf).resolve()
            try:
                candidate.relative_to(base.resolve())
            except ValueError:
                continue
            if candidate.is_dir():
                return str(candidate)

    return None


def preview_mlx_weights_source(model: str, explicit_model_dir: str | None) -> str:
    """Where weights are expected on disk (for UI); may not exist yet for fresh HF pulls."""
    raw = (model or "").strip()
    got = _resolve_non_hf_disk_path(raw, explicit_model_dir)
    if got is not None:
        return got
    if looks_like_hf_repo_id(raw):
        return str(hf_local_weights_directory(raw, explicit_model_dir))
    return raw


def resolve_mlx_weights_directory(model: str, explicit_model_dir: str | None) -> str:
    """Resolve ``model`` and optional ``explicit_model_dir`` to an on-disk MLX weights tree."""
    raw = (model or "").strip()
    disk = _resolve_non_hf_disk_path(raw, explicit_model_dir)
    if disk is not None:
        return disk

    if looks_like_hf_repo_id(raw):
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise RuntimeError(
                "huggingface_hub is required to download MLX weights from Hugging Face. "
                "Install with:  pip install huggingface_hub\n"
                "Or use a local directory for --model."
            ) from e
        dest = hf_local_weights_directory(raw, explicit_model_dir)
        dest.mkdir(parents=True, exist_ok=True)
        if _model_snapshot_present(dest):
            log.info("Using existing local MLX snapshot for %r at %s", raw, dest)
            return str(dest)
        log.info(
            "Ensuring Hugging Face weights %r under %s "
            "(huggingface_hub.snapshot_download; same payload as `huggingface-cli download`) …",
            raw,
            dest,
        )
        _snapshot_download_weights(snapshot_download, raw, dest)
        return str(dest)

    return raw


def _spill_slug(prompt: str, maxlen: int = 48) -> str:
    s = re.sub(r"[^\w\s-]+", "", prompt.lower().strip())[:maxlen]
    s = re.sub(r"[\s_]+", "_", s).strip("_")
    return s or "clip"


def _largest_mp4_under(root: Path) -> Path | None:
    best: Path | None = None
    best_mtime = -1.0
    try:
        for p in root.rglob("*.mp4"):
            try:
                st = p.stat()
            except OSError:
                continue
            if st.st_size <= 0:
                continue
            if st.st_mtime >= best_mtime:
                best_mtime = st.st_mtime
                best = p
    except OSError:
        return None
    return best


def _align_ltx2_spatial(n: int, align: int = LTX2_SPATIAL_ALIGN) -> int:
    if n < align:
        return align
    lower = (n // align) * align
    upper = lower + align
    return lower if (n - lower) <= (upper - n) else upper


def _nearest_valid_frames(n: int) -> int:
    if n < 9:
        return 9
    remainder = (n - 1) % 8
    if remainder == 0:
        return n
    lower = n - remainder
    upper = lower + 8
    return lower if (n - lower) <= (upper - n) else upper


def _decode_initial_image_dict(image_data: dict) -> str:
    """Data URL / path / base64 → path or URL (same contract as ``server._decode_initial_image``)."""
    data_url: str = (image_data.get("data_url") or "").strip()
    if data_url.startswith(("http://", "https://")):
        return data_url
    if data_url.startswith("file://"):
        from urllib.parse import unquote
        from urllib.request import url2pathname

        path = url2pathname(unquote(data_url[7:]))
        if os.path.isfile(path):
            return path
    if data_url and os.path.isfile(data_url):
        return data_url

    if data_url.startswith("data:"):
        header, encoded = data_url.split(",", 1)
        mime = header.split(";")[0].split(":")[1]
    else:
        mime = image_data.get("mime_type", "image/jpeg")
        encoded = data_url

    ext = mimetypes.guess_extension(mime) or ".jpg"
    if ext == ".jpe":
        ext = ".jpg"

    fd, path = tempfile.mkstemp(suffix=ext, prefix="fvserver_img_")
    with os.fdopen(fd, "wb") as f:
        f.write(base64.b64decode(encoded))
    return path


def _download_remote_to_temp(
    url: str,
    prefix: str,
    suffix_hint: str = "",
    max_bytes: int | None = MAX_REMOTE_INPUT_BYTES,
) -> str:
    req_url = (url or "").strip()
    if not req_url.startswith(("http://", "https://")):
        raise ValueError(f"Unsupported remote input URL: {url!r}")
    with urlopen(req_url, timeout=180) as resp:
        if max_bytes is None:
            payload = resp.read()
        else:
            payload = resp.read(max_bytes + 1)
    if max_bytes is not None and len(payload) > max_bytes:
        raise RuntimeError(
            f"Remote media exceeds {max_bytes // (1024 * 1024)} MiB limit"
        )
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix_hint)
    with os.fdopen(fd, "wb") as f:
        f.write(payload)
    return path


def _local_lora_cache_dir() -> Path:
    env = (os.environ.get(VIDEOFENTANYL_LORA_DIR_ENV) or "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (REPO_ROOT / "loras").resolve()


def _normalize_lora_spec(spec: str) -> str:
    """Normalize common Hugging Face URL variants to resolve/download form."""
    raw = (spec or "").strip()
    if not raw or not raw.startswith(("http://", "https://")):
        return raw
    parsed = urlparse(raw)
    host = parsed.netloc.lower()
    if host in ("hf.co", "www.hf.co"):
        raw = f"https://huggingface.co{parsed.path}"
        if parsed.query:
            raw += f"?{parsed.query}"
    if "huggingface.co" in raw and "/blob/" in raw:
        raw = raw.replace("/blob/", "/resolve/", 1)
    return raw


def _pick_safetensors_file(root: Path) -> Path | None:
    candidates = sorted(root.rglob("*.safetensors"))
    if not candidates:
        return None
    # Prefer explicit loras/ subdir when present.
    for c in candidates:
        if "loras" in {p.lower() for p in c.parts}:
            return c
    return candidates[0]


def _lora_cached_path(spec: str) -> Path | None:
    """Return local path when spec is already on disk; None if download may be needed."""
    raw = _normalize_lora_spec(spec)
    if not raw:
        return None

    p = Path(raw).expanduser()
    if p.is_file():
        return p.resolve()

    if raw.startswith(("http://", "https://")):
        parsed = urlparse(raw)
        if parsed.netloc.endswith("huggingface.co") and "/resolve/" in parsed.path:
            parts = [part for part in parsed.path.strip("/").split("/") if part]
            if len(parts) >= 5 and parts[2] == "resolve":
                repo_id = f"{parts[0]}/{parts[1]}"
                revision = parts[3]
                filename = "/".join(parts[4:])
                cache_root = _local_lora_cache_dir()
                local_dir = cache_root / repo_id.replace("/", "__")
                candidate = local_dir / filename
                if candidate.is_file():
                    return candidate.resolve()
                # hf_hub_download may also use hub cache layout under local_dir
                if local_dir.is_dir():
                    for match in local_dir.rglob(Path(filename).name):
                        if match.is_file():
                            return match.resolve()

    if looks_like_hf_repo_id(raw):
        dest = (_local_lora_cache_dir() / raw.replace("/", "__")).resolve()
        if dest.is_dir():
            picked = _pick_safetensors_file(dest)
            if picked is not None:
                return picked.resolve()

    return None


def _resolve_lora_path(spec: str) -> tuple[str, str | None]:
    """
    Resolve LoRA spec to a local safetensors path.
    Returns (path, cleanup_temp_path_or_none).
    """
    raw = _normalize_lora_spec(spec)
    if not raw:
        raise ValueError("Empty LoRA spec")

    cached = _lora_cached_path(raw)
    if cached is not None:
        log.debug("Using cached LoRA at %s", cached)
        return str(cached), None

    p = Path(raw).expanduser()
    if p.is_file():
        return str(p.resolve()), None
    if raw.startswith(("http://", "https://")):
        parsed = urlparse(raw)
        # Support Hugging Face resolve URLs directly by routing through hf_hub_download,
        # which handles large files and cache efficiently.
        if parsed.netloc.endswith("huggingface.co") and "/resolve/" in parsed.path:
            parts = [p for p in parsed.path.strip("/").split("/") if p]
            # Expected: <repo_owner>/<repo_name>/resolve/<revision>/<filename...>
            if len(parts) >= 5 and parts[2] == "resolve":
                repo_id = f"{parts[0]}/{parts[1]}"
                revision = parts[3]
                filename = "/".join(parts[4:])
                try:
                    from huggingface_hub import hf_hub_download
                except ImportError as e:
                    raise RuntimeError(
                        "huggingface_hub is required to download LoRA from Hugging Face"
                    ) from e
                cache_root = _local_lora_cache_dir()
                cache_root.mkdir(parents=True, exist_ok=True)
                log.info(
                    "Downloading LoRA %s (%s @ %s) …",
                    repo_id,
                    filename,
                    revision,
                )
                local = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    revision=revision,
                    local_dir=str(cache_root / repo_id.replace("/", "__")),
                )
                return str(Path(local).resolve()), None

        # Generic URL fallback (no 512MiB cap for LoRA artifacts).
        tmp = _download_remote_to_temp(
            raw,
            "fvserver_lora_",
            ".safetensors",
            max_bytes=None,
        )
        return tmp, tmp

    if looks_like_hf_repo_id(raw):
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise RuntimeError(
                "huggingface_hub is required to download LoRA from Hugging Face"
            ) from e
        dest_root = _local_lora_cache_dir()
        dest = (dest_root / raw.replace("/", "__")).resolve()
        dest.mkdir(parents=True, exist_ok=True)
        snap = _snapshot_download_weights(snapshot_download, raw, dest)
        snap_path = Path(snap)
        lora_file = _pick_safetensors_file(snap_path)
        if lora_file is None:
            raise RuntimeError(f"No .safetensors LoRA file found under {snap_path}")
        return str(lora_file.resolve()), None

    raise FileNotFoundError(f"LoRA spec not found or unsupported: {raw}")


def _decode_media_input(
    media_data: dict | str | None,
    *,
    temp_prefix: str,
    default_suffix: str,
) -> tuple[str | None, str | None]:
    """
    Resolve media input to a local path or URL.

    Returns: (resolved_path_or_url, temp_file_to_cleanup_or_none)
    """
    if media_data is None:
        return None, None

    if isinstance(media_data, str):
        raw = media_data.strip()
        if not raw:
            return None, None
        if raw.startswith(("http://", "https://")):
            tmp = _download_remote_to_temp(raw, temp_prefix, default_suffix)
            return tmp, tmp
        if raw.startswith("file://"):
            path = url2pathname(unquote(raw[7:]))
            if os.path.isfile(path):
                return path, None
            raise FileNotFoundError(f"File URL does not exist: {raw}")
        if os.path.isfile(raw):
            return raw, None
        raise FileNotFoundError(f"Media input not found: {raw}")

    if isinstance(media_data, dict):
        data_url = str(media_data.get("data_url") or "").strip()
        if not data_url:
            return None, None
        name_hint = str(
            media_data.get("name") or media_data.get("filename") or ""
        ).strip()
        name_suffix = Path(name_hint).suffix if name_hint else ""
        if data_url.startswith(("http://", "https://")):
            tmp = _download_remote_to_temp(data_url, temp_prefix, default_suffix)
            return tmp, tmp
        if data_url.startswith("file://"):
            path = url2pathname(unquote(data_url[7:]))
            if os.path.isfile(path):
                return path, None
            raise FileNotFoundError(f"File URL does not exist: {data_url}")
        if os.path.isfile(data_url):
            return data_url, None
        if data_url.startswith("data:"):
            header, encoded = data_url.split(",", 1)
            mime = header.split(";")[0].split(":")[1]
        else:
            mime = str(media_data.get("mime_type") or "")
            encoded = data_url
        ext = name_suffix or mimetypes.guess_extension(mime) or default_suffix
        if ext == ".jpe":
            ext = ".jpg"
        fd, path = tempfile.mkstemp(prefix=temp_prefix, suffix=ext)
        with os.fdopen(fd, "wb") as f:
            f.write(base64.b64decode(encoded))
        return path, path

    return None, None


def _decode_weighted_media_inputs(
    items: list[tuple[dict | str, float]] | None,
    *,
    temp_prefix: str,
    default_suffix: str,
) -> tuple[list[tuple[str, float]], list[str]]:
    decoded: list[tuple[str, float]] = []
    temps: list[str] = []
    for src, weight in (items or []):
        path, cleanup = _decode_media_input(
            src,
            temp_prefix=temp_prefix,
            default_suffix=default_suffix,
        )
        if path:
            decoded.append((path, float(weight)))
        if cleanup:
            temps.append(cleanup)
    return decoded, temps


def _apply_pending_loras(pipe: Any, lora_paths: list[tuple[str, float]] | None) -> None:
    if hasattr(pipe, "_pending_loras"):
        pipe._pending_loras = list(lora_paths or [])


def _pipeline_load_state_inconsistent(pipe: Any) -> bool:
    """True when ``_loaded`` is set but core weights the next job needs were freed."""
    if not getattr(pipe, "_loaded", False):
        return False
    if getattr(pipe, "dit", None) is None:
        return True
    if getattr(pipe, "vae_encoder", None) is None:
        return True
    return False


def _sync_pipeline_load_flag(pipe: Any) -> None:
    """Clear ``_loaded`` when freed blocks would make :meth:`load` skip a required reload."""
    if _pipeline_load_state_inconsistent(pipe):
        pipe._loaded = False


def _release_pipe_after_generation(pipe: Any) -> None:
    """Reset per-request state on a cached pipeline instance.

    Always clears request-scoped LoRA specs (``_pending_loras``) so the next job
    does not inherit the previous clip's adapters.

    When ``low_memory`` is off, leaves DiT / VAE encoder / decoders loaded so
    back-to-back generations reuse warm weights and fused LoRAs.

    When ``low_memory`` is on, upstream ``generate_and_save`` / decode paths
    already free blocks between stages; we only reconcile ``_loaded`` if a
    partial free left the flag set (which would skip ``load()`` and crash on
    ``assert self.vae_encoder is not None`` on the next job).
    """
    if hasattr(pipe, "_pending_loras"):
        pipe._pending_loras = []
    _sync_pipeline_load_flag(pipe)


def _unlink_fvserver_temp(path: str | None, marker: str) -> None:
    if path and os.path.isfile(path) and marker in path:
        try:
            os.unlink(path)
        except OSError:
            pass


def _export_output_mp4(source_path: str) -> str:
    """Copy generation output to a standalone temp file (outside per-job workdirs)."""
    fd, final_path = tempfile.mkstemp(prefix="fvserver_out_", suffix=".mp4")
    os.close(fd)
    shutil.copy2(source_path, final_path)
    return final_path


def _normalize_pipeline_profile(raw: str | None) -> str:
    profile = (raw or PIPE_PROFILE_DISTILLED).strip().lower()
    if profile in VALID_PIPELINE_PROFILES:
        return profile
    return PIPE_PROFILE_DISTILLED


def _maybe_enhance_prompt(
    prompt: str,
    *,
    mode: str,
    model_dir: str,
    enabled: bool,
) -> str:
    """Run ltx-2-mlx Gemma prompt enhancement when available and requested."""
    text = (prompt or "").strip()
    if not enabled or not text:
        return prompt
    enhance_mode = "i2v" if mode in ("generate", "i2v", "keyframe") else "t2v"
    try:
        import ltx_pipelines_mlx as lpm
    except ImportError:
        log.warning("enhance_prompt requested but ltx_pipelines_mlx is not installed")
        return prompt
    for attr in ("enhance_prompt", "enhance"):
        fn = getattr(lpm, attr, None)
        if callable(fn):
            try:
                out = fn(text, mode=enhance_mode, model_dir=model_dir)
                if isinstance(out, str) and out.strip():
                    log.info("Prompt enhanced via ltx_pipelines_mlx.%s", attr)
                    return out.strip()
            except TypeError:
                try:
                    out = fn(text, enhance_mode, model_dir)
                    if isinstance(out, str) and out.strip():
                        log.info("Prompt enhanced via ltx_pipelines_mlx.%s (legacy signature)", attr)
                        return out.strip()
                except Exception as exc:
                    log.warning("Prompt enhance via %s failed: %s", attr, exc)
            except Exception as exc:
                log.warning("Prompt enhance via %s failed: %s", attr, exc)
    log.warning(
        "enhance_prompt requested but no enhance API found in ltx_pipelines_mlx; using original prompt"
    )
    return prompt


def _apply_optional_generate_kwargs(call_kwargs: dict[str, Any], req: GenerationRequest) -> None:
    """Attach optional CFG / stage-2 / audio-regen flags when the pipeline accepts them."""
    if req.cfg_scale is not None:
        call_kwargs["cfg_scale"] = float(req.cfg_scale)
    if req.stg_scale is not None:
        call_kwargs["stg_scale"] = float(req.stg_scale)
    if req.stage2_steps is not None:
        call_kwargs["stage2_steps"] = int(req.stage2_steps)
    if req.no_regen_audio:
        call_kwargs["no_regen_audio"] = True
    if req.reference_strength is not None:
        call_kwargs["reference_strength"] = float(req.reference_strength)


def _frame_rate_from_kwargs(kwargs: dict[str, Any], default: float) -> float:
    if "frame_rate" in kwargs:
        return float(kwargs.pop("frame_rate"))
    if "fps" in kwargs:
        return float(kwargs.pop("fps"))
    return float(default)


def _decode_latents_to_mp4(
    pipe: Any,
    video_latent: Any,
    audio_latent: Any,
    output_path: str,
    frame_rate: float,
) -> None:
    if getattr(pipe, "low_memory", False):
        pipe.dit = None
        if hasattr(pipe, "prompt_encoder"):
            pipe.prompt_encoder.free()
        if hasattr(pipe, "image_conditioner"):
            pipe.image_conditioner.free()
        _sync_pipeline_load_flag(pipe)
        try:
            from ltx_core_mlx.utils.memory import aggressive_cleanup

            aggressive_cleanup()
        except ImportError:
            pass
    pipe._load_decoders()
    fn = getattr(pipe, "_decode_and_save_video", None)
    if fn is None:
        raise RuntimeError(f"{type(pipe).__name__} has no _decode_and_save_video()")
    sig = inspect.signature(fn)
    accepted = set(sig.parameters.keys())
    decode_kwargs: dict[str, Any] = {}
    if "frame_rate" in accepted:
        decode_kwargs["frame_rate"] = float(frame_rate)
    elif "fps" in accepted:
        decode_kwargs["fps"] = float(frame_rate)
    fn(video_latent, audio_latent, output_path, **decode_kwargs)


def _filter_call_kwargs(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    sig = inspect.signature(fn)
    accepted = set(sig.parameters.keys())
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_varkw:
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in accepted}


def _invoke_retake_and_save(pipe: Any, *, default_fps: float, **kwargs: Any) -> None:
    output_path = kwargs.pop("output_path")
    frame_rate = _frame_rate_from_kwargs(kwargs, default_fps)
    lora_paths = kwargs.pop("lora_paths", None)
    for drop_key in ("height", "width", "num_frames"):
        kwargs.pop(drop_key, None)
    _apply_pending_loras(pipe, lora_paths)
    video_latent, audio_latent = pipe.retake_from_video(
        **_filter_call_kwargs(pipe.retake_from_video, kwargs)
    )
    _decode_latents_to_mp4(pipe, video_latent, audio_latent, output_path, frame_rate)


def _invoke_extend_and_save(pipe: Any, *, default_fps: float, **kwargs: Any) -> None:
    output_path = kwargs.pop("output_path")
    frame_rate = _frame_rate_from_kwargs(kwargs, default_fps)
    lora_paths = kwargs.pop("lora_paths", None)
    for drop_key in ("height", "width", "num_frames"):
        kwargs.pop(drop_key, None)
    _apply_pending_loras(pipe, lora_paths)
    video_latent, audio_latent = pipe.extend_from_video(
        **_filter_call_kwargs(pipe.extend_from_video, kwargs)
    )
    _decode_latents_to_mp4(pipe, video_latent, audio_latent, output_path, frame_rate)


def _invoke_generate_and_save(pipe: Any, **kwargs: Any) -> None:
    """
    Call ``pipe.generate_and_save`` while tolerating API drift between ltx-2-mlx versions.

    - Drops unsupported kwargs.
    - Maps ``num_steps`` -> ``stage1_steps`` / ``steps`` when needed.
    - Maps ``fps`` -> ``frame_rate`` when upstream uses that name.
    - Applies request LoRAs via ``pipe._pending_loras`` when supported.
    """
    fn = getattr(pipe, "generate_and_save", None)
    if fn is None:
        raise RuntimeError(f"{type(pipe).__name__} has no generate_and_save()")

    sig = inspect.signature(fn)
    params = sig.parameters
    accepted = set(params.keys())
    has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())

    call_kwargs = dict(kwargs)
    lora_paths = call_kwargs.pop("lora_paths", None)
    _apply_pending_loras(pipe, lora_paths)

    if "num_steps" in call_kwargs:
        steps = call_kwargs["num_steps"]
        if "stage1_steps" in accepted and "stage1_steps" not in call_kwargs:
            call_kwargs["stage1_steps"] = steps
        if "num_steps" not in accepted and "steps" in accepted:
            call_kwargs["steps"] = call_kwargs.pop("num_steps")
    if "fps" in call_kwargs and "fps" not in accepted and "frame_rate" in accepted:
        call_kwargs["frame_rate"] = float(call_kwargs.pop("fps"))
    elif "fps" in call_kwargs and "fps" not in accepted and "frame_rate" not in accepted:
        call_kwargs.pop("fps", None)
    if "frame_rate" in call_kwargs and "frame_rate" not in accepted and "fps" in accepted:
        call_kwargs["fps"] = float(call_kwargs.pop("frame_rate"))

    img = call_kwargs.get("image")
    if img and "image" not in accepted:
        for alias in (
            "image_path",
            "input_image",
            "reference_image",
            "init_image",
            "first_frame_image",
            "start_image",
            "start",
        ):
            if alias in accepted:
                call_kwargs[alias] = call_kwargs.pop("image")
                break

    end_img = call_kwargs.get("end_image")
    if end_img and "end_image" not in accepted:
        for alias in ("end_image_path", "end", "target_image", "last_frame_image"):
            if alias in accepted:
                call_kwargs[alias] = call_kwargs.pop("end_image")
                break

    vid = call_kwargs.get("video_path") or call_kwargs.get("reference_video")
    if vid:
        for primary, aliases in (
            ("video_path", ("video", "source_video", "source_video_path", "input_video")),
            ("reference_video", ("video_path", "video", "source_video")),
        ):
            if primary in call_kwargs and primary not in accepted:
                for alias in aliases:
                    if alias in accepted:
                        call_kwargs[alias] = call_kwargs.pop(primary)
                        break

    if not has_varkw:
        dropped_image = img and "image" not in call_kwargs and not any(
            k in call_kwargs for k in ("image_path", "input_image", "reference_image", "init_image")
        )
        if dropped_image:
            log.warning(
                "Pipeline %s.generate_and_save does not accept image= — I2V conditioning disabled",
                type(pipe).__name__,
            )
        call_kwargs = {k: v for k, v in call_kwargs.items() if k in accepted}

    fn(**call_kwargs)


def _mux_audio_into_video(
    video_path: str,
    audio_path: str,
    output_path: str,
    *,
    duration_s: float,
) -> None:
    """Mux an audio track into a silent video (a2v chain visual continuation)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "ffmpeg is required to mux audio into a2v autocontinue clips"
        )
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-t",
        f"{max(0.1, duration_s):.6f}",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        output_path,
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        err = (cp.stderr or cp.stdout or "unknown ffmpeg error").strip()
        raise RuntimeError(f"ffmpeg audio mux failed: {err}")


class _ModelProgressStore:
    """Thread-safe denoising / download progress for WebSocket keepalives."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, Any] | None = None

    def set(self, data: dict[str, Any]) -> None:
        with self._lock:
            self._data = dict(data)

    def clear(self) -> None:
        with self._lock:
            self._data = None

    def snapshot(self) -> dict[str, Any] | None:
        with self._lock:
            if not self._data:
                return None
            snap = dict(self._data)
        step = snap.get("step")
        total = snap.get("total")
        if (
            snap.get("pct") is None
            and isinstance(step, (int, float))
            and isinstance(total, (int, float))
            and total > 0
        ):
            snap["pct"] = round(100 * float(step) / float(total), 0)
        return snap


class GenerationCancelledError(RuntimeError):
    """Raised when generation is cancelled via ``request_cancel()``."""


def _stage_from_tqdm_desc(desc: str) -> str:
    d = (desc or "").strip().lower()
    if "denois" in d:
        return "denoising"
    if "download" in d:
        return "downloading"
    if any(k in d for k in ("encod", "decod", "vae", "latent")):
        return "encoding"
    if "upscal" in d:
        return "upscaling"
    return "generating"


class LocalVideoGenerator:
    """
    MLX pipeline adapter for ``ltx-2-mlx``: text/image/audio/video generation modes.
    Weights are resolved once at ``load()``; individual pipelines lazy-load on demand.
    """

    def __init__(
        self,
        model: str,
        num_frames: int,
        height: int,
        width: int,
        fps: float,
        model_dir: str | None,
        inference_steps: int,
        default_lora_specs: list[tuple[str, float]] | None = None,
        spill_dir: Path | None = None,
        low_memory: bool = False,
        *,
        upscale: bool = False,
    ) -> None:
        self.model = model
        self.num_frames = int(num_frames)
        self.height = int(height)
        self.width = int(width)
        self.fps = float(fps)
        self.model_dir = model_dir
        self.inference_steps = max(1, int(inference_steps))
        self.default_lora_specs = list(default_lora_specs or [])
        self.spill_dir = spill_dir
        self.low_memory = bool(low_memory)
        # Backward-compatible ctor arg used by server.py CLI.
        self.upscale = bool(upscale)
        self._model_path: str | None = None
        self._pipe_classes: dict[str, Any] = {}
        self._pipes: dict[str, Any] = {}
        self._resolved_default_loras: list[tuple[str, float]] | None = None
        self._lpm_module: Any | None = None
        self._model_progress = _ModelProgressStore()
        self._cancel_requested = threading.Event()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="ltx-gen",
        )
        self._executor_shutdown = False

    def shutdown(self, *, wait: bool = False) -> None:
        """Release the generation thread pool (call on server exit)."""
        self.request_cancel()
        if self._executor_shutdown:
            return
        self._executor.shutdown(wait=wait, cancel_futures=True)
        self._executor_shutdown = True

    def clear_cancel(self) -> None:
        self._cancel_requested.clear()

    def request_cancel(self) -> None:
        self._cancel_requested.set()

    def _check_cancel(self) -> None:
        if self._cancel_requested.is_set():
            raise GenerationCancelledError("Generation cancelled")

    @contextmanager
    def _track_model_progress(self):
        """Patch tqdm so denoising step bars update ``model_progress_for_ws``."""
        try:
            import tqdm as tqdm_mod
        except ImportError:
            yield
            return

        generator = self
        orig_tqdm = tqdm_mod.tqdm
        orig_auto = getattr(tqdm_mod.auto, "tqdm", orig_tqdm)

        class _TrackingTqdm(orig_tqdm):  # type: ignore[misc,valid-type]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                generator._check_cancel()
                super().__init__(*args, **kwargs)
                self._publish(generator)

            def __iter__(self):
                for item in super().__iter__():
                    generator._check_cancel()
                    yield item

            def refresh(self, *args: Any, **kwargs: Any) -> None:
                generator._check_cancel()
                super().refresh(*args, **kwargs)
                self._publish(generator)

            def update(self, n: float = 1) -> bool | None:
                generator._check_cancel()
                result = super().update(n)
                self._publish(generator)
                return result

            def _publish(self, gen: LocalVideoGenerator) -> None:
                desc = str(self.desc or "")
                fd = getattr(self, "format_dict", None) or {}
                n = int(self.n)
                total = int(self.total) if self.total is not None else None
                rate = fd.get("rate")
                tqdm_elapsed = fd.get("elapsed")
                eta_s: float | None = None
                avg_step_s: float | None = None
                if isinstance(rate, (int, float)) and rate > 0:
                    avg_step_s = round(1.0 / float(rate), 2)
                    if total is not None:
                        eta_s = round((total - n) / float(rate), 1)
                gen._model_progress.set(
                    {
                        "stage": _stage_from_tqdm_desc(desc),
                        "step": n,
                        "total": total,
                        "eta_s": eta_s,
                        "avg_step_s": avg_step_s,
                        "elapsed_s": (
                            round(float(tqdm_elapsed), 1)
                            if isinstance(tqdm_elapsed, (int, float))
                            else None
                        ),
                        "label": desc.strip() or None,
                    }
                )

        tqdm_mod.tqdm = _TrackingTqdm
        tqdm_mod.auto.tqdm = _TrackingTqdm
        samplers_mod: Any | None = None
        orig_samplers_tqdm: Any = None
        try:
            import ltx_pipelines_mlx.utils.samplers as samplers_mod

            orig_samplers_tqdm = getattr(samplers_mod, "tqdm", None)
            samplers_mod.tqdm = _TrackingTqdm
        except ImportError:
            pass
        try:
            yield
        finally:
            tqdm_mod.tqdm = orig_tqdm
            tqdm_mod.auto.tqdm = orig_auto
            if samplers_mod is not None and orig_samplers_tqdm is not None:
                samplers_mod.tqdm = orig_samplers_tqdm
            self._model_progress.clear()

    def _resolve_model_dir(self) -> str:
        return resolve_mlx_weights_directory(self.model, self.model_dir)

    def load(self) -> None:
        if self._model_path is not None:
            return
        try:
            import ltx_pipelines_mlx as lpm
        except ImportError as e:
            raise RuntimeError(
                "Missing ltx_pipelines_mlx. Install the MLX monorepo packages, e.g.:\n"
                f"{ltx2_mlx_install_hint()}"
            ) from e
        path = self._resolve_model_dir()
        self._model_path = path
        self._lpm_module = lpm

        generate_cls = getattr(lpm, "DistilledPipeline", None)
        if generate_cls is None:
            generate_cls = getattr(lpm, "TextToVideoPipeline", None)
        if self.upscale:
            upscale_cls = getattr(lpm, "TI2VidTwoStagesPipeline", None)
            if upscale_cls is not None:
                generate_cls = upscale_cls
                log.info("Using TI2VidTwoStagesPipeline for --upscale generate jobs")

        legacy_t2v_cls = getattr(lpm, "TextToVideoPipeline", None)
        legacy_i2v_cls = getattr(lpm, "ImageToVideoPipeline", None)

        a2v_cls = getattr(lpm, "A2VidPipelineTwoStage", None)
        if a2v_cls is None:
            a2v_cls = getattr(lpm, "AudioToVideoPipeline", None)

        retake_cls = getattr(lpm, "RetakePipeline", None)
        extend_cls = retake_cls if retake_cls is not None else getattr(lpm, "ExtendPipeline", None)

        self._pipe_classes: dict[str, Any] = {}
        if legacy_t2v_cls is not None:
            self._pipe_classes["t2v"] = legacy_t2v_cls
        elif generate_cls is not None:
            self._pipe_classes["t2v"] = generate_cls
        if legacy_i2v_cls is not None:
            self._pipe_classes["i2v"] = legacy_i2v_cls
            log.info("Using ImageToVideoPipeline for i2v / autocontinue conditioning")
        else:
            one_stage_i2v_cls = getattr(lpm, "TI2VidOneStagePipeline", None)
            if one_stage_i2v_cls is not None:
                self._pipe_classes["i2v"] = one_stage_i2v_cls
                log.info(
                    "Using TI2VidOneStagePipeline for i2v / autocontinue conditioning"
                )
            elif generate_cls is not None:
                self._pipe_classes["i2v"] = generate_cls
        if generate_cls is not None:
            self._pipe_classes["gen"] = generate_cls
        if a2v_cls is not None:
            self._pipe_classes["a2v"] = a2v_cls
        if retake_cls is not None:
            self._pipe_classes["retake"] = retake_cls
        if extend_cls is not None:
            self._pipe_classes["extend"] = extend_cls

        ic_cls = getattr(lpm, "ICLoraPipeline", None)
        if ic_cls is not None:
            self._pipe_classes["ic_lora"] = ic_cls

        for key, cls_name in (
            ("two_stage", "TI2VidTwoStagesPipeline"),
            ("hq", "TI2VidTwoStagesHQPipeline"),
            ("keyframe", "KeyframePipeline"),
            ("lipdub", "LipDubPipeline"),
        ):
            cls = getattr(lpm, cls_name, None)
            if cls is not None:
                self._pipe_classes[key] = cls
                log.info("Registered MLX pipeline %s (%s)", key, cls_name)

        # Legacy standalone spatial upscaler classes (pre-v0.14 monolith pipelines).
        for cls_name in (
            "SpatialUpscalerX2V11Pipeline",
            "SpatialUpscalerX2Pipeline",
            "SpatialUpscalerPipeline",
            "LTXSpatialUpscalerPipeline",
        ):
            up_cls = getattr(lpm, cls_name, None)
            if up_cls is not None:
                self._pipe_classes["spatial_upscaler"] = up_cls
                log.info("Detected spatial upscaler pipeline class: %s", cls_name)
                break
        log.info("MLX model path resolved ✓ %s", path)

    def _get_pipe(self, key: str, *, pipe_kwargs: dict[str, Any] | None = None) -> Any:
        if not pipe_kwargs and key in self._pipes:
            return self._pipes[key]
        self.load()
        if self._model_path is None:
            raise RuntimeError("MLX model path not initialized")
        cls = self._pipe_classes.get(key)
        if cls is None:
            raise RuntimeError(
                f"Unsupported pipeline key: {key} (installed ltx-2-mlx may be too old; "
                f"expected {LTX2_MLX_GIT_TAG}+)"
            )
        log.info("Loading MLX pipeline %s from %s …", key, self._model_path)
        ctor_kwargs: dict[str, Any] = {"model_dir": self._model_path, "low_memory": self.low_memory}
        if pipe_kwargs:
            ctor_kwargs.update(pipe_kwargs)
        pipe = cls(**ctor_kwargs)
        if key not in ("retake", "extend") and hasattr(pipe, "load"):
            pipe.load()
        if not pipe_kwargs:
            self._pipes[key] = pipe
        log.info("MLX pipeline ready ✓ (%s)", key)
        return pipe

    def _resolve_generate_pipe_key(self, profile: str, *, has_image: bool) -> str:
        profile = _normalize_pipeline_profile(profile)
        if profile == PIPE_PROFILE_TWO_STAGE and "two_stage" in self._pipe_classes:
            return "two_stage"
        if profile == PIPE_PROFILE_HQ and "hq" in self._pipe_classes:
            return "hq"
        if profile == PIPE_PROFILE_ONE_STAGE:
            if has_image and "i2v" in self._pipe_classes:
                return "i2v"
            return "t2v"
        if self.upscale and "two_stage" in self._pipe_classes:
            return "two_stage"
        if has_image and "i2v" in self._pipe_classes:
            return "i2v"
        return "t2v"

    def _resolve_lora_specs(self, specs: list[tuple[str, float]]) -> tuple[list[tuple[str, float]], list[str]]:
        resolved: list[tuple[str, float]] = []
        temps: list[str] = []
        for lora_spec, lora_scale in specs:
            lora_path, cleanup = _resolve_lora_path(str(lora_spec))
            resolved.append((lora_path, float(lora_scale)))
            if cleanup:
                temps.append(cleanup)
        return resolved, temps

    def ensure_default_loras_ready(self) -> None:
        """
        Resolve/download default LoRAs at startup when LoRA mode is enabled.
        """
        self.load()
        if not self.default_lora_specs:
            self._resolved_default_loras = []
            return
        resolved, temps = self._resolve_lora_specs(self.default_lora_specs)
        for tmp in temps:
            if tmp and os.path.isfile(tmp) and "fvserver_lora_" in tmp:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
        self._resolved_default_loras = resolved
        log.info("Resolved %d default LoRA(s) for global use", len(resolved))

    def model_progress_for_ws(self) -> dict[str, Any] | None:
        return self._model_progress.snapshot()

    def default_lora_count(self) -> int:
        if self._resolved_default_loras is not None:
            return len(self._resolved_default_loras)
        return len(self.default_lora_specs)

    def _calculate_stage1_dimensions(self, height: int, width: int) -> tuple[int, int]:
        base_h = _align_ltx2_spatial(max(LTX2_SPATIAL_ALIGN, int(round(height / 2.0))))
        base_w = _align_ltx2_spatial(max(LTX2_SPATIAL_ALIGN, int(round(width / 2.0))))
        return base_h, base_w

    def _run_spatial_upscaler_stage(
        self,
        *,
        prompt: str,
        source_video_path: str,
        output_path: str,
        height: int,
        width: int,
        num_frames: int,
        seed: int,
        num_steps: int,
        lora_paths: list[tuple[str, float]],
    ) -> bool:
        try:
            pipe = self._get_pipe("spatial_upscaler")
        except Exception as exc:
            log.warning(
                "Spatial upscaler pipeline unavailable; using first-stage output only: %s",
                exc,
            )
            return False

        try:
            sig = inspect.signature(pipe.generate_and_save)
            accepted = set(sig.parameters.keys())
            call_kwargs: dict[str, Any] = {
                "prompt": prompt,
                "output_path": output_path,
                "num_frames": num_frames,
                "fps": float(self.fps),
                "seed": seed,
                "num_steps": num_steps,
                "lora_paths": lora_paths,
            }
            # Stage-2 source size comes from source_video_path; these are output target dimensions.
            if "target_height" in accepted and "target_width" in accepted:
                call_kwargs["target_height"] = height
                call_kwargs["target_width"] = width
            elif "height" in accepted and "width" in accepted:
                call_kwargs["height"] = height
                call_kwargs["width"] = width

            # Backend compatibility: select only one supported input-video arg name.
            for name in (
                "video",
                "video_path",
                "source_video",
                "source_video_path",
                "input_video",
                "input_video_path",
            ):
                if name in accepted:
                    call_kwargs[name] = source_video_path
                    break

            # Backend compatibility: pick the first recognized control by preference:
            # explicit boolean flags first, then string sampler-name style controls.
            for name, value in (
                ("use_tiled_sampler", True),
                ("tiled", True),
                ("sampler", "tiled"),
                ("sampler_name", "tiled"),
                ("sampling_method", "tiled"),
                ("second_sampler", "tiled"),
            ):
                if name in accepted:
                    call_kwargs[name] = value
                    break

            _invoke_generate_and_save(
                pipe,
                **call_kwargs,
            )
        except Exception as exc:
            log.warning(
                "Spatial upscaler second stage failed; using first-stage output only: %s",
                exc,
            )
            return False

        out = Path(output_path)
        if not (out.is_file() and out.stat().st_size > 0):
            log.warning(
                "Spatial upscaler produced no output; using first-stage output only: %s",
                output_path,
            )
            return False
        return True

    async def generate(
        self,
        prompt: str,
        image_data: dict | str | None = None,
        audio_data: dict | str | None = None,
        source_video_data: dict | str | None = None,
        seed: int = -1,
        num_frames: int | None = None,
        height: int | None = None,
        width: int | None = None,
        negative_prompt: str = "",
        mode: str = "generate",
        num_steps: int | None = None,
        retake_start: int | None = None,
        retake_end: int | None = None,
        extend_frames: int | None = None,
        extend_direction: str = "after",
        lora_specs: list[tuple[str, float]] | None = None,
        video_conditioning_specs: list[tuple[dict | str, float]] | None = None,
        *,
        job_id: str | None = None,
        a2v_visual_i2v_continue: bool = False,
        end_image_data: dict | str | None = None,
        enhance_prompt: bool = False,
        pipeline_profile: str = PIPE_PROFILE_DISTILLED,
        cfg_scale: float | None = None,
        stg_scale: float | None = None,
        stage2_steps: int | None = None,
        no_regen_audio: bool = False,
        reference_strength: float | None = None,
    ) -> str:
        self.clear_cancel()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            functools.partial(
                self._generate_sync,
                GenerationRequest(
                    prompt=prompt,
                    image_data=image_data,
                    audio_data=audio_data,
                    source_video_data=source_video_data,
                    seed=seed,
                    num_frames=num_frames or self.num_frames,
                    height=height or self.height,
                    width=width or self.width,
                    negative_prompt=negative_prompt,
                    mode=mode or "generate",
                    num_steps=num_steps,
                    retake_start=retake_start,
                    retake_end=retake_end,
                    extend_frames=extend_frames,
                    extend_direction=extend_direction or "after",
                    lora_specs=lora_specs,
                    video_conditioning_specs=video_conditioning_specs,
                    job_id=job_id,
                    a2v_visual_i2v_continue=a2v_visual_i2v_continue,
                    end_image_data=end_image_data,
                    enhance_prompt=enhance_prompt,
                    pipeline_profile=pipeline_profile,
                    cfg_scale=cfg_scale,
                    stg_scale=stg_scale,
                    stage2_steps=stage2_steps,
                    no_regen_audio=no_regen_audio,
                    reference_strength=reference_strength,
                ),
            ),
        )

    def _salvage_mp4_to_spill(
        self,
        tmpdir: str,
        preferred_out: str,
        job_id: str | None,
        prompt: str,
        tag: str,
    ) -> None:
        if not self.spill_dir or not job_id:
            return
        root = Path(tmpdir)
        src = Path(preferred_out)
        if not (src.is_file() and src.stat().st_size > 0):
            alt = _largest_mp4_under(root)
            if alt is None:
                log.warning(
                    "  ◆ no MP4 found to salvage under %s (job %s)",
                    tmpdir,
                    job_id[:8],
                )
                return
            src = alt
        try:
            self.spill_dir.mkdir(parents=True, exist_ok=True)
            slug = _spill_slug(prompt)
            dest = self.spill_dir / f"{job_id}_{slug}_{tag}.mp4"
            shutil.copy2(src, dest)
            log.info("  ◆ spill-salvaged (%s) → %s", tag, dest)
        except OSError as exc:
            log.error("  ✗ spill salvage failed: %s", exc)

    def _generate_sync(self, req: GenerationRequest) -> str:
        del req.negative_prompt  # reserved for future CFG-enabled variants
        self._check_cancel()
        self.load()
        self._check_cancel()

        assert self._model_path is not None
        requested_height = int(req.height or self.height)
        requested_width = int(req.width or self.width)
        ah = _align_ltx2_spatial(requested_height)
        aw = _align_ltx2_spatial(requested_width)
        if ah != requested_height or aw != requested_width:
            log.warning(
                "LTX requires H×W divisible by %s; adjusted %s×%s → %s×%s",
                LTX2_SPATIAL_ALIGN,
                requested_height,
                requested_width,
                ah,
                aw,
            )
        height, width = ah, aw

        requested_num_frames = int(req.num_frames or self.num_frames)
        nf = _nearest_valid_frames(requested_num_frames)
        if nf != requested_num_frames:
            log.warning(
                "LTX requires (frames-1)%%8==0; adjusted frames %s → %s",
                requested_num_frames,
                nf,
            )
        mode = (req.mode or "generate").strip().lower()
        requested_steps = int(req.num_steps or self.inference_steps)
        steps = max(1, requested_steps)
        if steps != requested_steps:
            log.warning("LTX steps must be >=1; adjusted steps %s → %s", requested_steps, steps)
        requested_seed = int(req.seed)
        seed = requested_seed
        if seed < 0:
            # videofentanyl commonly sends -1 for "auto/random seed".
            seed = random.randint(0, 2**31 - 1)
            log.info("LTX random seed requested (%s); using generated seed %s", requested_seed, seed)
        effective_loras: list[tuple[str, float]] = []
        if self._resolved_default_loras is not None:
            effective_loras.extend(self._resolved_default_loras)
        else:
            effective_loras.extend(self.default_lora_specs)
        effective_loras.extend(req.lora_specs or [])
        resolved_loras: list[tuple[str, float]] = []

        tmp_image: str | None = None
        tmp_end_image: str | None = None
        tmp_audio: str | None = None
        tmp_video: str | None = None
        tmp_video_conditioning_cleanup: list[str] = []
        tmp_lora_cleanup: list[str] = []
        prefix = f"fv_{req.job_id[:8]}_" if req.job_id else "fvserver_work_"
        tmpdir = tempfile.mkdtemp(prefix=prefix)
        out_path = os.path.join(tmpdir, "output.mp4")
        last_pipe: Any | None = None
        media_cleanups: list[str] = []

        try:
            tmp_image, tmp_image_cleanup = _decode_media_input(
                req.image_data,
                temp_prefix="fvserver_img_",
                default_suffix=".jpg",
            )
            if not tmp_image and isinstance(req.image_data, dict):
                tmp_image = _decode_initial_image_dict(req.image_data)
                tmp_image_cleanup = tmp_image
            tmp_audio, tmp_audio_cleanup = _decode_media_input(
                req.audio_data,
                temp_prefix="fvserver_audio_",
                default_suffix=".wav",
            )
            tmp_video, tmp_video_cleanup = _decode_media_input(
                req.source_video_data,
                temp_prefix="fvserver_video_",
                default_suffix=".mp4",
            )
            tmp_end_image, tmp_end_image_cleanup = _decode_media_input(
                req.end_image_data,
                temp_prefix="fvserver_end_img_",
                default_suffix=".jpg",
            )
            if not tmp_end_image and isinstance(req.end_image_data, dict):
                tmp_end_image = _decode_initial_image_dict(req.end_image_data)
                tmp_end_image_cleanup = tmp_end_image
            vc_items, vc_cleanup = _decode_weighted_media_inputs(
                req.video_conditioning_specs,
                temp_prefix="fvserver_vcond_",
                default_suffix=".mp4",
            )
            tmp_video_conditioning_cleanup = vc_cleanup
            for path, cleanup, marker in (
                (tmp_image, tmp_image_cleanup, "fvserver_img_"),
                (tmp_end_image, tmp_end_image_cleanup, "fvserver_end_img_"),
                (tmp_audio, tmp_audio_cleanup, "fvserver_audio_"),
                (tmp_video, tmp_video_cleanup, "fvserver_video_"),
            ):
                if cleanup:
                    media_cleanups.append(cleanup)
                elif path and marker in path:
                    media_cleanups.append(path)
            if self._resolved_default_loras is not None and not req.lora_specs:
                resolved_loras = list(self._resolved_default_loras)
            else:
                for lora_spec, lora_scale in effective_loras:
                    lora_path, lora_cleanup = _resolve_lora_path(str(lora_spec))
                    resolved_loras.append((lora_path, float(lora_scale)))
                    if lora_cleanup:
                        tmp_lora_cleanup.append(lora_cleanup)
            effective_prompt = _maybe_enhance_prompt(
                req.prompt,
                mode=mode,
                model_dir=str(self._model_path),
                enabled=bool(req.enhance_prompt) and mode not in ("extend", "retake"),
            )
            profile = _normalize_pipeline_profile(req.pipeline_profile)
            log.info(
                "Generation effective params: mode=%s profile=%s enhance=%s seed=%s (requested=%s) "
                "size=%sx%s frames=%s steps=%s fps=%s (requested size=%sx%s frames=%s steps=%s) "
                "image=%s end_image=%s audio=%s video=%s retake=%s-%s extend=%s/%s vcond=%s loras=%s "
                "model_path=%s",
                mode,
                profile if mode not in ("extend", "retake") else "dev+CFG",
                "yes" if req.enhance_prompt else "no",
                seed,
                requested_seed,
                height,
                width,
                nf,
                steps,
                float(self.fps),
                requested_height,
                requested_width,
                requested_num_frames,
                requested_steps,
                "yes" if tmp_image else "no",
                "yes" if tmp_end_image else "no",
                "yes" if tmp_audio else "no",
                "yes" if tmp_video else "no",
                req.retake_start if req.retake_start is not None else "-",
                req.retake_end if req.retake_end is not None else "-",
                req.extend_frames if req.extend_frames is not None else "-",
                (req.extend_direction or "after").strip().lower(),
                len(vc_items),
                len(resolved_loras),
                self._model_path,
            )
            if resolved_loras:
                log.info(
                    "Applying %d LoRA(s) for mode=%s (request=%d, defaults=%d)",
                    len(resolved_loras),
                    mode,
                    len(req.lora_specs or []),
                    self.default_lora_count(),
                )

            self._check_cancel()
            try:
                with self._track_model_progress():
                    common_gen_kwargs = dict(
                        prompt=effective_prompt,
                        output_path=out_path,
                        height=height,
                        width=width,
                        num_frames=nf,
                        frame_rate=float(self.fps),
                        seed=seed,
                        num_steps=steps,
                        lora_paths=resolved_loras,
                    )
                    _apply_optional_generate_kwargs(common_gen_kwargs, req)
                    if mode == "a2v":
                        if not tmp_audio:
                            raise RuntimeError("a2v mode requires audio input")
                        video_duration_s = nf / float(self.fps)
                        if req.a2v_visual_i2v_continue and tmp_image:
                            log.info(
                                "A2V chain continue: i2v visual + audio mux "
                                "(avoids A2V re-conditioning on autocontinue frame)"
                            )
                            silent_path = os.path.join(tmpdir, "output_silent.mp4")
                            pipe = self._get_pipe("i2v")
                            last_pipe = pipe
                            _invoke_generate_and_save(
                                pipe,
                                **common_gen_kwargs,
                                output_path=silent_path,
                                image=tmp_image,
                            )
                            _mux_audio_into_video(
                                silent_path,
                                tmp_audio,
                                out_path,
                                duration_s=video_duration_s,
                            )
                        else:
                            pipe = self._get_pipe("a2v")
                            last_pipe = pipe
                            _invoke_generate_and_save(
                                pipe,
                                **common_gen_kwargs,
                                audio_path=tmp_audio,
                                image=tmp_image,
                            )
                    elif mode == "retake":
                        if not tmp_video:
                            raise RuntimeError("retake mode requires source video input")
                        start_frame = int(req.retake_start if req.retake_start is not None else 1)
                        end_frame = int(req.retake_end if req.retake_end is not None else start_frame)
                        pipe = self._get_pipe("retake")
                        last_pipe = pipe
                        retake_steps = steps
                        retake_cfg = float(
                            req.cfg_scale
                            if req.cfg_scale is not None
                            else RETAKE_EXTEND_DEFAULT_CFG
                        )
                        retake_stg = float(
                            req.stg_scale
                            if req.stg_scale is not None
                            else RETAKE_EXTEND_DEFAULT_STG
                        )
                        retake_kwargs = dict(
                            prompt=effective_prompt,
                            output_path=out_path,
                            video_path=tmp_video,
                            start_frame=start_frame,
                            end_frame=end_frame,
                            seed=seed,
                            num_steps=retake_steps,
                            cfg_scale=retake_cfg,
                            stg_scale=retake_stg,
                            lora_paths=resolved_loras,
                            fps=float(self.fps),
                        )
                        _apply_optional_generate_kwargs(retake_kwargs, req)
                        if not callable(getattr(pipe, "retake_from_video", None)):
                            raise RuntimeError(
                                f"{type(pipe).__name__} has no retake_from_video(); "
                                "update ltx-2-mlx"
                            )
                        log.info(
                            "Retake via retake_from_video (frames %s-%s, steps=%s, cfg=%.1f, stg=%.1f)",
                            start_frame,
                            end_frame,
                            retake_steps,
                            retake_cfg,
                            retake_stg,
                        )
                        _invoke_retake_and_save(
                            pipe,
                            default_fps=float(self.fps),
                            **retake_kwargs,
                        )
                    elif mode == "extend":
                        if not tmp_video:
                            raise RuntimeError("extend mode requires source video input")
                        ext_frames = int(req.extend_frames if req.extend_frames is not None else 2)
                        direction = (req.extend_direction or "after").strip().lower()
                        pipe = self._get_pipe("extend")
                        last_pipe = pipe
                        extend_steps = steps
                        extend_cfg = float(
                            req.cfg_scale
                            if req.cfg_scale is not None
                            else RETAKE_EXTEND_DEFAULT_CFG
                        )
                        extend_stg = float(
                            req.stg_scale
                            if req.stg_scale is not None
                            else RETAKE_EXTEND_DEFAULT_STG
                        )
                        extend_kwargs = dict(
                            prompt=effective_prompt,
                            output_path=out_path,
                            video_path=tmp_video,
                            extend_frames=ext_frames,
                            direction=direction,
                            seed=seed,
                            num_steps=extend_steps,
                            cfg_scale=extend_cfg,
                            stg_scale=extend_stg,
                            lora_paths=resolved_loras,
                            fps=float(self.fps),
                        )
                        _apply_optional_generate_kwargs(extend_kwargs, req)
                        if not callable(getattr(pipe, "extend_from_video", None)):
                            raise RuntimeError(
                                f"{type(pipe).__name__} has no extend_from_video(); "
                                "update ltx-2-mlx"
                            )
                        log.info(
                            "Extend via extend_from_video "
                            "(extend_frames=%s, direction=%s, steps=%s, cfg=%.1f, stg=%.1f)",
                            ext_frames,
                            direction,
                            extend_steps,
                            extend_cfg,
                            extend_stg,
                        )
                        _invoke_extend_and_save(
                            pipe,
                            default_fps=float(self.fps),
                            **extend_kwargs,
                        )
                        try:
                            from videofentanyl import count_video_frames

                            src_frames = count_video_frames(tmp_video) if tmp_video else None
                            out_frames = count_video_frames(out_path)
                            if src_frames is not None and out_frames is not None:
                                log.info(
                                    "Extend output: %d frames (source %d, +%d, ~%.2fs @ %.1f fps)",
                                    out_frames,
                                    src_frames,
                                    out_frames - src_frames,
                                    max(0.0, (out_frames - 1) / float(self.fps)),
                                    float(self.fps),
                                )
                                if out_frames <= src_frames:
                                    log.warning(
                                        "Extend did not lengthen the video — verify duration "
                                        "(5s = 121 frames) and extend_frames=%s",
                                        ext_frames,
                                    )
                        except Exception:
                            pass
                    elif mode == "keyframe":
                        if not tmp_image or not tmp_end_image:
                            raise RuntimeError("keyframe mode requires start and end images")
                        pipe = self._get_pipe("keyframe")
                        last_pipe = pipe
                        _invoke_generate_and_save(
                            pipe,
                            **common_gen_kwargs,
                            image=tmp_image,
                            end_image=tmp_end_image,
                        )
                    elif mode in ("lipdub", "lip_dub"):
                        if not tmp_video:
                            raise RuntimeError("lipdub mode requires reference video")
                        if len(resolved_loras) != 1:
                            raise RuntimeError("lipdub mode requires exactly one LoRA spec")
                        pipe = self._get_pipe(
                            "lipdub",
                            pipe_kwargs={
                                "lora_paths": [(str(p), float(s)) for p, s in resolved_loras],
                            },
                        )
                        last_pipe = pipe
                        _invoke_generate_and_save(
                            pipe,
                            **common_gen_kwargs,
                            video_path=tmp_video,
                            reference_video=tmp_video,
                        )
                    elif mode == "ic_lora":
                        if not resolved_loras:
                            raise RuntimeError("ic_lora mode requires at least one LoRA spec")
                        if not vc_items:
                            raise RuntimeError("ic_lora mode requires video_conditioning entries")
                        pipe = self._get_pipe(
                            "ic_lora",
                            pipe_kwargs={
                                "lora_paths": [(str(p), float(s)) for p, s in resolved_loras],
                            },
                        )
                        last_pipe = pipe
                        _invoke_generate_and_save(
                            pipe,
                            prompt=req.prompt,
                            output_path=out_path,
                            video_conditioning=[(str(p), float(s)) for p, s in vc_items],
                            height=height,
                            width=width,
                            num_frames=nf,
                            fps=float(self.fps),
                            seed=seed,
                            num_steps=steps,
                        )
                    elif tmp_image:
                        try:
                            from PIL import Image as PILImage

                            with PILImage.open(tmp_image) as im:
                                log.info(
                                    "I2V conditioning image: %s (%dx%d) → generation %dx%d",
                                    tmp_image,
                                    im.size[0],
                                    im.size[1],
                                    width,
                                    height,
                                )
                        except Exception:
                            log.info(
                                "I2V conditioning image: %s → generation %dx%d",
                                tmp_image,
                                width,
                                height,
                            )
                        # Separate i2v/two-stage instance: do not reuse the t2v pipe cache entry.
                        pipe_key = self._resolve_generate_pipe_key(profile, has_image=True)
                        pipe = self._get_pipe(pipe_key)
                        last_pipe = pipe
                        _invoke_generate_and_save(
                            pipe,
                            **common_gen_kwargs,
                            image=tmp_image,
                        )
                    else:
                        pipe_key = self._resolve_generate_pipe_key(profile, has_image=False)
                        pipe = self._get_pipe(pipe_key)
                        last_pipe = pipe
                        if (
                            profile == PIPE_PROFILE_DISTILLED
                            and self.upscale
                            and "spatial_upscaler" in self._pipe_classes
                            and pipe_key == "t2v"
                        ):
                            base_h, base_w = self._calculate_stage1_dimensions(height, width)
                            lowres_out_path = os.path.join(tmpdir, "output_lowres.mp4")
                            log.info(
                                "Legacy two-stage upscale enabled: stage1=%sx%s -> stage2=%sx%s",
                                base_h,
                                base_w,
                                height,
                                width,
                            )
                            _invoke_generate_and_save(
                                pipe,
                                prompt=req.prompt,
                                output_path=lowres_out_path,
                                height=base_h,
                                width=base_w,
                                num_frames=nf,
                                fps=float(self.fps),
                                seed=seed,
                                num_steps=steps,
                                lora_paths=resolved_loras,
                            )
                            upscaled = self._run_spatial_upscaler_stage(
                                prompt=req.prompt,
                                source_video_path=lowres_out_path,
                                output_path=out_path,
                                height=height,
                                width=width,
                                num_frames=nf,
                                seed=seed,
                                num_steps=steps,
                                lora_paths=resolved_loras,
                            )
                            if not upscaled:
                                shutil.copy2(lowres_out_path, out_path)
                        else:
                            _invoke_generate_and_save(
                                pipe,
                                **common_gen_kwargs,
                            )
            except BaseException as exc:
                if not isinstance(exc, GenerationCancelledError):
                    log.exception(
                        "Generation failed (job %s, mode=%s): %s",
                        req.job_id[:8] if req.job_id else "?",
                        mode,
                        exc,
                    )
                    self._salvage_mp4_to_spill(
                        tmpdir, out_path, req.job_id, req.prompt, "ENCODE_FAIL"
                    )
                raise

            video_path = out_path
            if not os.path.exists(video_path):
                self._salvage_mp4_to_spill(
                    tmpdir, out_path, req.job_id, req.prompt, "MISSING_OUTPUT",
                )
                raise RuntimeError(
                    f"Generation completed but output file not found: {video_path}"
                )
            if last_pipe is not None:
                _release_pipe_after_generation(last_pipe)
            return _export_output_mp4(video_path)

        finally:
            for tmp in media_cleanups:
                _unlink_fvserver_temp(tmp, "fvserver_")
            for tmp in tmp_video_conditioning_cleanup:
                _unlink_fvserver_temp(tmp, "fvserver_vcond_")
            for tmp in tmp_lora_cleanup:
                _unlink_fvserver_temp(tmp, "fvserver_lora_")
            shutil.rmtree(tmpdir, ignore_errors=True)
