"""Tests for vendored VideoDecoderPruna + Hub / local PrunaVAED MLX weights."""

from __future__ import annotations

import mlx.core as mx
import mlx.utils

from ltx_mlx_backend import PRUNA_VAED_HF_REPO, ensure_pruna_vae_decoder_files
from ltx_video_decoder_pruna import VideoDecoderPruna


def test_pruna_hub_repo_constant():
    assert PRUNA_VAED_HF_REPO == "audiohacking/pruna-vaed-mlx"


def test_pruna_decoder_param_topology():
    dec = VideoDecoderPruna()
    keys = {k for k, _ in mlx.utils.tree_flatten(dec.parameters())}
    assert "up_blocks.1.conv_in.conv_shortcut.weight" in keys
    assert "up_blocks.2.resnets.5.conv1.conv.weight" in keys
    w = dict(mlx.utils.tree_flatten(dec.parameters()))
    assert tuple(w["conv_out.conv.weight"].shape) == (48, 3, 3, 3, 64)


def test_ensure_pruna_resolves_local_or_hub():
    weights, cfg = ensure_pruna_vae_decoder_files()
    assert weights.is_file()
    assert weights.name == "vae_decoder_pruna.safetensors"
    assert cfg is None or cfg.is_file()


def test_load_converted_weights_and_decode():
    from ltx_core_mlx.utils.weights import load_split_safetensors

    weights_path, cfg_path = ensure_pruna_vae_decoder_files()
    dec = (
        VideoDecoderPruna.from_config(cfg_path)
        if cfg_path is not None
        else VideoDecoderPruna()
    )
    weights = load_split_safetensors(weights_path, prefix="vae_decoder.")
    model_keys = {k for k, _ in mlx.utils.tree_flatten(dec.parameters())}
    missing = sorted(model_keys - set(weights))
    assert not missing, f"missing keys: {missing[:10]}"
    dec.load_weights(list(weights.items()), strict=True)
    out = dec.decode(mx.random.normal((1, 128, 5, 8, 8)))
    mx.eval(out)
    assert out.shape[0] == 1 and out.shape[1] == 3
    assert out.shape[3] == 256 and out.shape[4] == 256
