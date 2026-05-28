#!/usr/bin/env python3
"""
mcp_server.py — MCP interface for local ltx-ws generation.

Exposes standardized MCP tools so any MCP client can drive the existing
WebSocket workflow implemented by ``server.py`` + ``videofentanyl.py``.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

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

DEFAULT_SERVER_URL = "ws://127.0.0.1:8765/ws"
DEFAULT_OUTPUT_DIR = "mcp_outputs"
DEFAULT_PREFIX = "ltx_mcp"

_SERVER_URL = DEFAULT_SERVER_URL
_OUTPUT_DIR = Path(DEFAULT_OUTPUT_DIR)
_VERBOSE = False

mcp = FastMCP("ltx-ws")


def _ts_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _new_output_path(prompt: str, output_dir: Path, prefix: str) -> Path:
    slug = sanitize_filename(prompt) or "clip"
    return output_dir / f"{prefix}_{slug}_{_ts_slug()}.mp4"


def _normalize_mode(mode: str) -> str:
    val = (mode or "generate").strip().lower()
    allowed = {"generate", "a2v", "retake", "extend", "ic_lora"}
    if val not in allowed:
        raise ValueError(f"Unsupported mode {mode!r}; expected one of {sorted(allowed)}")
    return val


def _build_params(
    *,
    prompt: str,
    mode: str,
    image: str | None,
    audio: str | None,
    video: str | None,
    seed: int | None,
    num_frames: int | None,
    height: int | None,
    width: int | None,
    num_steps: int | None,
    retake_start: int | None,
    retake_end: int | None,
    extend_frames: int | None,
    extend_direction: str | None,
    lora_specs: list[list[Any]] | None,
    video_conditioning: list[list[Any]] | None,
) -> GenerationParams:
    normalized_mode = _normalize_mode(mode)

    image_payload = load_image_payload(image) if image else None
    audio_payload = load_media_payload(audio, kind="audio") if audio else None
    video_payload = load_media_payload(video, kind="video") if video else None

    parsed_loras: list[tuple[str, float]] = []
    for item in lora_specs or []:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("Each lora_specs item must be [path_or_repo_or_url, scale]")
        parsed_loras.append((str(item[0]).strip(), float(item[1])))

    parsed_vcond: list[tuple[dict, float]] = []
    for item in video_conditioning or []:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("Each video_conditioning item must be [video_path_or_url, scale]")
        payload = load_media_payload(str(item[0]).strip(), kind="video")
        parsed_vcond.append((payload, float(item[1])))

    return GenerationParams(
        prompt=prompt.strip(),
        preset_id="simple_custom_prompt",
        enhancement_enabled=False,
        single_clip_mode=True,
        auto_extension_enabled=False,
        loop_generation_enabled=False,
        initial_image=image_payload,
        seed=seed,
        num_frames=num_frames,
        height=height,
        width=width,
        num_steps=num_steps,
        generation_mode=normalized_mode,
        audio_input=audio_payload,
        source_video=video_payload,
        retake_start=retake_start,
        retake_end=retake_end,
        extend_frames=extend_frames,
        extend_direction=extend_direction,
        lora_specs=parsed_loras,
        video_conditioning_specs=parsed_vcond,
    )


async def _run_job(params: GenerationParams, output_path: Path) -> dict[str, Any]:
    job = Job(
        id=1,
        params=params,
        output_path=output_path,
        max_attempts=1,
    )
    job.status = JobStatus.RUNNING
    job.started_at = time.time()

    session = VideoSession(job=job, mode="ltx", verbose=_VERBOSE)
    ok = await session.run(idle_timeout=None)

    job.finished_at = time.time()
    job.status = JobStatus.DONE if ok else JobStatus.FAILED

    if not ok:
        raise RuntimeError(job.error or "Generation failed")

    return {
        "output_path": str(output_path.resolve()),
        "bytes": int(job.file_bytes),
        "chunks": int(job.chunk_count),
        "segments": int(job.segment_count),
        "elapsed_s": round(job.elapsed, 3),
        "ttff_ms": job.ttff_ms,
        "generation_ms": job.gen_latency_ms,
        "e2e_ms": job.e2e_latency_ms,
    }


def _build_multi_job(
    *,
    job_id: int,
    prompt: str,
    mode: str,
    image_payload: dict | None,
    audio_payload: dict | None,
    video_payload: dict | None,
    seed: int | None,
    num_frames: int | None,
    height: int | None,
    width: int | None,
    num_steps: int | None,
    retake_start: int | None,
    retake_end: int | None,
    extend_frames: int | None,
    extend_direction: str | None,
    lora_specs: list[tuple[str, float]],
    video_conditioning_specs: list[tuple[dict, float]],
    output_path: Path,
) -> Job:
    params = GenerationParams(
        prompt=prompt.strip(),
        preset_id="simple_custom_prompt",
        enhancement_enabled=False,
        single_clip_mode=True,
        auto_extension_enabled=False,
        loop_generation_enabled=False,
        initial_image=image_payload,
        seed=seed,
        num_frames=num_frames,
        height=height,
        width=width,
        num_steps=num_steps,
        generation_mode=mode,
        audio_input=audio_payload,
        source_video=video_payload,
        retake_start=retake_start,
        retake_end=retake_end,
        extend_frames=extend_frames,
        extend_direction=extend_direction,
        lora_specs=lora_specs,
        video_conditioning_specs=video_conditioning_specs,
    )
    return Job(
        id=job_id,
        params=params,
        output_path=output_path,
        max_attempts=1,
    )


@mcp.tool()
async def ltx_generate_video(
    prompt: str,
    mode: str = "generate",
    image: str | None = None,
    audio: str | None = None,
    video: str | None = None,
    seed: int | None = None,
    num_frames: int | None = None,
    height: int | None = None,
    width: int | None = None,
    num_steps: int | None = None,
    retake_start: int | None = None,
    retake_end: int | None = None,
    extend_frames: int | None = None,
    extend_direction: str | None = None,
    lora_specs: list[list[Any]] | None = None,
    video_conditioning: list[list[Any]] | None = None,
    output_filename: str | None = None,
) -> dict[str, Any]:
    """
    Generate a single video clip through ltx-ws and return file/latency metadata.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt is required")

    if mode == "a2v" and not audio:
        raise ValueError("mode=a2v requires audio")
    if mode == "retake":
        if not video:
            raise ValueError("mode=retake requires video")
        if retake_start is None or retake_end is None:
            raise ValueError("mode=retake requires retake_start and retake_end")
    if mode == "extend":
        if not video:
            raise ValueError("mode=extend requires video")
        if extend_frames is None:
            raise ValueError("mode=extend requires extend_frames")
    if mode == "ic_lora":
        if not lora_specs:
            raise ValueError("mode=ic_lora requires lora_specs")
        if not video_conditioning:
            raise ValueError("mode=ic_lora requires video_conditioning")

    params = _build_params(
        prompt=prompt,
        mode=mode,
        image=image,
        audio=audio,
        video=video,
        seed=seed,
        num_frames=num_frames,
        height=height,
        width=width,
        num_steps=num_steps,
        retake_start=retake_start,
        retake_end=retake_end,
        extend_frames=extend_frames,
        extend_direction=extend_direction,
        lora_specs=lora_specs,
        video_conditioning=video_conditioning,
    )

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if output_filename:
        output_path = _OUTPUT_DIR / output_filename
    else:
        output_path = _new_output_path(prompt, _OUTPUT_DIR, DEFAULT_PREFIX)

    # Route VideoSession to local ltx-ws endpoint configured at startup.
    import videofentanyl as vf_mod

    previous = vf_mod._SERVER_OVERRIDE
    vf_mod._SERVER_OVERRIDE = _SERVER_URL
    try:
        result = await _run_job(params=params, output_path=output_path)
    finally:
        vf_mod._SERVER_OVERRIDE = previous

    return {
        "ok": True,
        "server": _SERVER_URL,
        **result,
    }


@mcp.tool()
async def ltx_generate_sequence(
    prompts: list[str],
    mode: str = "generate",
    autocontinue: bool = True,
    autoconcat: bool = False,
    image: str | None = None,
    audio: str | None = None,
    video: str | None = None,
    seed: int | None = None,
    num_frames: int | None = None,
    height: int | None = None,
    width: int | None = None,
    num_steps: int | None = None,
    retake_start: int | None = None,
    retake_end: int | None = None,
    extend_frames: int | None = None,
    extend_direction: str | None = None,
    lora_specs: list[list[Any]] | None = None,
    video_conditioning: list[list[Any]] | None = None,
    output_prefix: str = DEFAULT_PREFIX,
) -> dict[str, Any]:
    """
    Generate multiple clips sequentially and optionally chain them via autocontinue.
    """
    clean_prompts = [p.strip() for p in prompts if isinstance(p, str) and p.strip()]
    if not clean_prompts:
        raise ValueError("prompts must contain at least one non-empty prompt")

    normalized_mode = _normalize_mode(mode)
    if normalized_mode == "a2v" and not audio:
        raise ValueError("mode=a2v requires audio")
    if normalized_mode == "retake":
        if not video:
            raise ValueError("mode=retake requires video")
        if retake_start is None or retake_end is None:
            raise ValueError("mode=retake requires retake_start and retake_end")
    if normalized_mode == "extend":
        if not video:
            raise ValueError("mode=extend requires video")
        if extend_frames is None:
            raise ValueError("mode=extend requires extend_frames")
    if normalized_mode == "ic_lora":
        if not lora_specs:
            raise ValueError("mode=ic_lora requires lora_specs")
        if not video_conditioning:
            raise ValueError("mode=ic_lora requires video_conditioning")

    image_payload = load_image_payload(image) if image else None
    audio_payload = load_media_payload(audio, kind="audio") if audio else None
    video_payload = load_media_payload(video, kind="video") if video else None

    parsed_loras: list[tuple[str, float]] = []
    for item in lora_specs or []:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("Each lora_specs item must be [path_or_repo_or_url, scale]")
        parsed_loras.append((str(item[0]).strip(), float(item[1])))

    parsed_vcond: list[tuple[dict, float]] = []
    for item in video_conditioning or []:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("Each video_conditioning item must be [video_path_or_url, scale]")
        payload = load_media_payload(str(item[0]).strip(), kind="video")
        parsed_vcond.append((payload, float(item[1])))

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs: list[Job] = []
    for i, p in enumerate(clean_prompts, start=1):
        clip_slug = sanitize_filename(p) or "clip"
        output_path = _OUTPUT_DIR / f"{output_prefix}_{i:03d}_{clip_slug}_{_ts_slug()}.mp4"
        jobs.append(
            _build_multi_job(
                job_id=i,
                prompt=p,
                mode=normalized_mode,
                image_payload=image_payload,
                audio_payload=audio_payload,
                video_payload=video_payload,
                seed=seed,
                num_frames=num_frames,
                height=height,
                width=width,
                num_steps=num_steps,
                retake_start=retake_start,
                retake_end=retake_end,
                extend_frames=extend_frames,
                extend_direction=extend_direction,
                lora_specs=parsed_loras,
                video_conditioning_specs=parsed_vcond,
                output_path=output_path,
            )
        )

    import videofentanyl as vf_mod

    previous = vf_mod._SERVER_OVERRIDE
    vf_mod._SERVER_OVERRIDE = _SERVER_URL
    started = time.time()
    try:
        results: list[dict[str, Any]] = []
        for idx, job in enumerate(jobs):
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            session = VideoSession(job=job, mode="ltx", verbose=_VERBOSE)
            ok = await session.run(idle_timeout=None)
            job.finished_at = time.time()
            job.status = JobStatus.DONE if ok else JobStatus.FAILED
            if not ok:
                raise RuntimeError(f"sequence failed at clip {idx + 1}: {job.error or 'generation failed'}")

            results.append(
                {
                    "index": idx + 1,
                    "prompt": job.params.prompt,
                    "output_path": str(job.output_path.resolve()),
                    "bytes": int(job.file_bytes),
                    "elapsed_s": round(job.elapsed, 3),
                    "ttff_ms": job.ttff_ms,
                    "generation_ms": job.gen_latency_ms,
                    "e2e_ms": job.e2e_latency_ms,
                }
            )

            if autocontinue and idx + 1 < len(jobs):
                next_frame = extract_last_frame(job.output_path)
                if not next_frame:
                    raise RuntimeError(f"autocontinue failed to extract last frame from clip {idx + 1}")
                jobs[idx + 1].params.initial_image = next_frame
        if autoconcat:
            # Reuse existing ffmpeg concat helper.
            await asyncio.to_thread(
                try_autoconcat_clips,
                jobs,
                output_prefix,
                "mp4",
                _VERBOSE,
                False,
            )
            merged_candidates = sorted(_OUTPUT_DIR.glob(f"{output_prefix}_merged_*.mp4"))
            merged_path = str(merged_candidates[-1].resolve()) if merged_candidates else None
        else:
            merged_path = None
    finally:
        vf_mod._SERVER_OVERRIDE = previous

    return {
        "ok": True,
        "server": _SERVER_URL,
        "count": len(results),
        "autocontinue": bool(autocontinue),
        "autoconcat": bool(autoconcat),
        "merged_output_path": merged_path,
        "total_elapsed_s": round(time.time() - started, 3),
        "clips": results,
    }


@mcp.tool()
async def ltx_server_healthcheck() -> dict[str, Any]:
    """
    Verify that the configured ltx-ws endpoint accepts a WebSocket connection.
    """
    import websockets

    started = time.time()
    try:
        async with websockets.connect(_SERVER_URL, open_timeout=10, close_timeout=5):
            pass
    except Exception as exc:
        return {
            "ok": False,
            "server": _SERVER_URL,
            "error": str(exc),
        }
    return {
        "ok": True,
        "server": _SERVER_URL,
        "latency_ms": int((time.time() - started) * 1000),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mcp_server",
        description="MCP server exposing ltx-ws generation tools",
    )
    p.add_argument("--server-url", default=DEFAULT_SERVER_URL, help="ltx-ws WebSocket URL")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="directory for generated videos")
    p.add_argument("--verbose", action="store_true", help="verbose WebSocket session logs")
    return p


def main() -> None:
    global _SERVER_URL, _OUTPUT_DIR, _VERBOSE
    args = _build_parser().parse_args()
    _SERVER_URL = args.server_url.strip()
    _OUTPUT_DIR = Path(args.output_dir).expanduser().resolve()
    _VERBOSE = bool(args.verbose)

    # FastMCP handles stdio transport by default.
    mcp.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
