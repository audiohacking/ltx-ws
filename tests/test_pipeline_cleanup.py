"""Pipeline memory cleanup between generation jobs."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ltx_mlx_backend import (
    LocalVideoGenerator,
    _free_pipeline_blocks,
    _pipeline_load_state_inconsistent,
    _release_pipe_after_generation,
    _sync_pipeline_load_flag,
)


class _Block:
    def __init__(self) -> None:
        self.freed = False

    def free(self) -> None:
        self.freed = True


def test_free_pipeline_blocks_clears_weights_and_blocks():
    pipe = MagicMock()
    pipe.dit = object()
    pipe.vae_encoder = object()
    pipe.prompt_encoder = _Block()
    pipe.image_conditioner = _Block()
    pipe.audio_conditioner = _Block()
    pipe.video_decoder_block = _Block()
    pipe.audio_decoder_block = _Block()
    pipe._loaded = True

    _free_pipeline_blocks(pipe)

    assert pipe.dit is None
    assert pipe.vae_encoder is None
    assert pipe.prompt_encoder.freed
    assert pipe.image_conditioner.freed
    assert pipe.audio_conditioner.freed
    assert pipe.video_decoder_block.freed
    assert pipe.audio_decoder_block.freed
    assert pipe._loaded is False


def test_pipeline_load_state_inconsistent_when_dit_freed():
    pipe = MagicMock()
    pipe._loaded = True
    pipe.dit = None
    pipe.vae_encoder = object()
    assert _pipeline_load_state_inconsistent(pipe) is True

    pipe.dit = object()
    pipe.vae_encoder = None
    assert _pipeline_load_state_inconsistent(pipe) is True

    pipe.vae_encoder = object()
    assert _pipeline_load_state_inconsistent(pipe) is False


def test_sync_pipeline_load_flag():
    pipe = MagicMock()
    pipe._loaded = True
    pipe.dit = None
    pipe.vae_encoder = object()
    _sync_pipeline_load_flag(pipe)
    assert pipe._loaded is False


def test_release_pipe_after_generation_clears_loras_and_blocks():
    pipe = MagicMock()
    pipe._pending_loras = [("x.safetensors", 1.0)]
    pipe.dit = object()
    pipe.vae_encoder = object()
    pipe.prompt_encoder = _Block()
    pipe.image_conditioner = None
    pipe.audio_conditioner = None
    pipe.video_decoder_block = None
    pipe.audio_decoder_block = None
    pipe._loaded = True

    with patch("ltx_mlx_backend._mlx_aggressive_cleanup") as cleanup:
        _release_pipe_after_generation(pipe)

    assert pipe._pending_loras == []
    assert pipe.dit is None
    assert pipe.prompt_encoder.freed
    cleanup.assert_called_once()


def test_cleanup_after_generation_with_pipe():
    gen = LocalVideoGenerator.__new__(LocalVideoGenerator)
    gen._model_progress = MagicMock()
    pipe = MagicMock()

    with patch("ltx_mlx_backend._release_pipe_after_generation") as release:
        gen.cleanup_after_generation(pipe)

    gen._model_progress.clear.assert_called_once()
    release.assert_called_once_with(pipe)


def test_cleanup_after_generation_without_pipe():
    gen = LocalVideoGenerator.__new__(LocalVideoGenerator)
    gen._model_progress = MagicMock()

    with patch("ltx_mlx_backend._mlx_aggressive_cleanup") as cleanup:
        gen.cleanup_after_generation()

    gen._model_progress.clear.assert_called_once()
    cleanup.assert_called_once()
