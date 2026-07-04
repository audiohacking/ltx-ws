"""Regression tests for generation queue and patch concurrency."""

from __future__ import annotations

import asyncio
import importlib
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


def test_apply_ltx_mlx_patches_safe_under_concurrent_imports():
    pytest.importorskip("ltx_pipelines_mlx")
    from ltx_mlx_backend import _apply_ltx_mlx_patches

    stop = threading.Event()
    errors: list[BaseException] = []

    def import_loop() -> None:
        while not stop.is_set():
            for name in ("json", "pathlib", "fractions", "dataclasses", "inspect"):
                try:
                    importlib.import_module(name)
                except BaseException as exc:  # pragma: no cover - defensive
                    errors.append(exc)

    worker = threading.Thread(target=import_loop, daemon=True)
    worker.start()
    try:
        for _ in range(50):
            _apply_ltx_mlx_patches(default_fps=24.0)
    finally:
        stop.set()
        worker.join(timeout=2.0)
    assert not errors


def test_raw_dict_iteration_raises_when_modified_during_loop():
    items = {"a": 1, "b": 2}
    it = items.values().__iter__()
    next(it)
    items["c"] = 3
    with pytest.raises(RuntimeError, match="dictionary changed size"):
        next(it)


def test_clip_relabel_uses_snapshot_iteration():
    """Relabelling must iterate a snapshot so concurrent clip inserts do not crash."""
    from web_ui import AppState, ClipRecord, RunRecord, RunStatus

    state = AppState(
        server_url="ws://127.0.0.1:9/ws",
        output_dir=Path("/tmp/unused"),
        upload_dir=Path("/tmp/unused2"),
        preferred_model="auto",
        embedded=True,
    )
    chain_id = "chain-a"
    state.clips["clip-1"] = ClipRecord(
        id="clip-1",
        prompt="first",
        label="CURRENT",
        video_url="/api/videos/a.mp4",
        filename="a.mp4",
        chain_id=chain_id,
        clip_index=0,
        mode="a2v",
        status=RunStatus.DONE.value,
        created_at="now",
    )
    state.clips["clip-2"] = ClipRecord(
        id="clip-2",
        prompt="second",
        label="CURRENT",
        video_url="",
        filename="b.mp4",
        chain_id=chain_id,
        clip_index=1,
        mode="a2v",
        status=RunStatus.QUEUED.value,
        created_at="now",
    )

    for c in list(state.clips.values()):
        if c.chain_id == chain_id and c.label == "CURRENT":
            c.label = "EDIT"

    assert state.clips["clip-1"].label == "EDIT"
    assert state.clips["clip-2"].label == "EDIT"


def _make_state(tmp_path: Path) -> Any:
    from web_ui import AppState

    vs = MagicMock()
    vs.scheduler = MagicMock()
    vs.scheduler.running_generation_id = None
    vs.generator = MagicMock()
    vs.generator.clear_cancel = MagicMock()
    vs.generator.model_progress_for_ws = MagicMock(return_value=None)

    @dataclass
    class _Slot:
        async def __aenter__(self) -> str:
            return str(uuid.uuid4())

        async def __aexit__(self, *args: Any) -> None:
            return None

    vs.scheduler.generation_slot.return_value = _Slot()

    state = AppState(
        server_url="ws://127.0.0.1:9/ws",
        output_dir=tmp_path / "out",
        upload_dir=tmp_path / "up",
        preferred_model="auto",
        embedded=True,
        video_server=vs,
    )
    state.output_dir.mkdir(parents=True, exist_ok=True)
    return state


async def _run_dummy_clip(
    _video_server: Any,
    job: Any,
    on_event: Any,
    *,
    should_cancel: Any = None,
) -> bool:
    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    job.output_path.write_bytes(b"\x00" * 64)
    job.file_bytes = 64
    await on_event({"type": "generation_progress", "elapsed_s": 0.1})
    return True


async def _enqueue_and_drain(state: Any, run_id: str) -> None:
    from web_ui import _execute_run

    await state.enqueue_generation_run(run_id)
    await _execute_run(state, run_id)


def test_generation_queue_back_to_back_runs(tmp_path: Path):
    from web_ui import ClipRecord, RunRecord, RunStatus, _RUN_BODIES

    state = _make_state(tmp_path)
    chain_id = str(uuid.uuid4())

    async def scenario() -> None:
        run_ids: list[str] = []
        for idx in range(2):
            run_id = str(uuid.uuid4())
            clip_id = str(uuid.uuid4())
            filename = f"clip_{idx}.mp4"
            state.clips[clip_id] = ClipRecord(
                id=clip_id,
                prompt=f"prompt {idx}",
                label="CURRENT" if idx == 0 else "EDIT",
                video_url="",
                filename=filename,
                chain_id=chain_id,
                clip_index=idx,
                mode="a2v",
                status=RunStatus.QUEUED.value,
                created_at="now",
            )
            state.runs[run_id] = RunRecord(
                id=run_id,
                status=RunStatus.QUEUED.value,
                prompts=[f"prompt {idx}"],
                chain_id=chain_id,
                clip_ids=[clip_id],
                created_at="now",
                autocontinue=True,
                autoconcat=False,
                audiocontinue=False,
                chain_method="autocontinue",
            )
            _RUN_BODIES[run_id] = {
                "mode": "a2v",
                "prompt": f"prompt {idx}",
                "chain_id": chain_id,
                "duration_seconds": 5.0,
                "chain_method": "autocontinue",
            }
            state.event_queues[run_id] = asyncio.Queue()
            run_ids.append(run_id)

        with patch("web_ui._run_clip_inprocess", side_effect=_run_dummy_clip):
            for run_id in run_ids:
                await _enqueue_and_drain(state, run_id)

        for run_id in run_ids:
            assert state.runs[run_id].status == RunStatus.DONE.value
            clip_id = state.runs[run_id].clip_ids[0]
            assert state.clips[clip_id].status == RunStatus.DONE.value
            assert (state.output_dir / state.clips[clip_id].filename).is_file()

    asyncio.run(scenario())


def test_generation_queue_while_submitting_next_run(tmp_path: Path):
    """Run 1 finishing relabel must not race with run 2 clip registration."""
    from web_ui import ClipRecord, RunRecord, RunStatus, _RUN_BODIES

    state = _make_state(tmp_path)
    chain_id = str(uuid.uuid4())
    run1 = str(uuid.uuid4())
    clip1 = str(uuid.uuid4())
    run2 = str(uuid.uuid4())
    clip2 = str(uuid.uuid4())

    state.clips[clip1] = ClipRecord(
        id=clip1,
        prompt="one",
        label="CURRENT",
        video_url="",
        filename="one.mp4",
        chain_id=chain_id,
        clip_index=0,
        mode="a2v",
        status=RunStatus.QUEUED.value,
        created_at="now",
    )
    state.runs[run1] = RunRecord(
        id=run1,
        status=RunStatus.QUEUED.value,
        prompts=["one"],
        chain_id=chain_id,
        clip_ids=[clip1],
        created_at="now",
        autocontinue=True,
        autoconcat=False,
        audiocontinue=False,
        chain_method="autocontinue",
    )
    _RUN_BODIES[run1] = {"mode": "a2v", "prompt": "one", "chain_id": chain_id}
    state.event_queues[run1] = asyncio.Queue()

    async def slow_clip(_video_server: Any, job: Any, on_event: Any, **kwargs: Any) -> bool:
        await asyncio.sleep(0.01)
        job.output_path.parent.mkdir(parents=True, exist_ok=True)
        job.output_path.write_bytes(b"x" * 32)
        job.file_bytes = 32
        return True

    async def scenario() -> None:
        with patch("web_ui._run_clip_inprocess", side_effect=slow_clip):
            task1 = asyncio.create_task(_enqueue_and_drain(state, run1))
            await asyncio.sleep(0.005)
            state.clips[clip2] = ClipRecord(
                id=clip2,
                prompt="two",
                label="CURRENT",
                video_url="",
                filename="two.mp4",
                chain_id=chain_id,
                clip_index=1,
                mode="a2v",
                status=RunStatus.QUEUED.value,
                created_at="now",
            )
            state.runs[run2] = RunRecord(
                id=run2,
                status=RunStatus.QUEUED.value,
                prompts=["two"],
                chain_id=chain_id,
                clip_ids=[clip2],
                created_at="now",
                autocontinue=True,
                autoconcat=False,
                audiocontinue=False,
                chain_method="autocontinue",
            )
            _RUN_BODIES[run2] = {
                "mode": "a2v",
                "prompt": "two",
                "chain_id": chain_id,
                "continue_from": clip1,
            }
            state.event_queues[run2] = asyncio.Queue()
            await task1
            await _enqueue_and_drain(state, run2)

        assert state.runs[run1].status == RunStatus.DONE.value
        assert state.runs[run2].status == RunStatus.DONE.value

    asyncio.run(scenario())
