"""PrunaVAED pruned LTX-2.3 video VAE decoder (diffusers topology).

Vendored for ltx-ws; same module lives in ltx-2-mlx as
``ltx_core_mlx.model.video_vae.video_decoder_pruna`` for upstream PRs.

Ports HuggingFace ``LTX2VideoDecoder3d`` + PrunaAI ``patch_diffusers.py``
wiring so pruned skip widths and ``conv_in`` projections load correctly.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.utils

from ltx_core_mlx.model.video_vae.convolution import Conv3dBlock
from ltx_core_mlx.model.video_vae.normalization import pixel_norm
from ltx_core_mlx.model.video_vae.ops import PerChannelStatistics
from ltx_core_mlx.model.video_vae.sampling import pixel_shuffle_3d, unpatchify_spatial

logger = logging.getLogger(__name__)

# Default PrunaVAED schedule (from PrunaAI/PrunaVAED vae/config.json)
_DEFAULT_BLOCK_OUT = (128, 256, 384, 1024)
_DEFAULT_LAYERS = (4, 6, 4, 2, 2)
_DEFAULT_UPSAMPLE_FACTOR = (2, 2, 1, 2)
_DEFAULT_UPSAMPLE_TYPE = ("spatiotemporal", "spatiotemporal", "temporal", "spatial")
_DEFAULT_UPSAMPLE_RESIDUAL = (False, False, False, False)
_DEFAULT_SPATIO_TEMPORAL = (True, True, True, True)

_STRIDE_BY_TYPE = {
    "spatial": (1, 2, 2),
    "temporal": (2, 1, 1),
    "spatiotemporal": (2, 2, 2),
}


class DiffusersResnet3d(nn.Module):
    """Diffusers LTX2VideoResnetBlock3d (PixelNorm + optional channel-change shortcut).

    Equal-channel path matches stock ``ResBlock3d`` (parameterless norms).
    Channel-change path adds ``norm3`` (LayerNorm) + ``conv_shortcut`` (1x1x1).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int | None = None,
        *,
        causal: bool = False,
        spatial_padding_mode: str = "zeros",
        eps: float = 1e-6,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv1 = Conv3dBlock(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            causal=causal,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.conv2 = Conv3dBlock(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            causal=causal,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.norm3: nn.LayerNorm | None = None
        self.conv_shortcut: nn.Conv3d | None = None
        if in_channels != out_channels:
            self.norm3 = nn.LayerNorm(dims=in_channels, eps=eps)
            # Plain 1x1x1 conv — key is conv_shortcut.{weight,bias} (not .conv.)
            self.conv_shortcut = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        h = self.conv1(nn.silu(pixel_norm(x)))
        h = self.conv2(nn.silu(pixel_norm(h)))
        if self.norm3 is not None:
            residual = self.norm3(residual)
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        return h + residual


class DiffusersUpsampler3d(nn.Module):
    """Diffusers LTX2VideoUpsampler3d (residual=False path).

    Conv expands channels, then depth-to-space rearrange; drops first
    ``stride[0]-1`` frames after temporal upsampling (matches diffusers).
    """

    def __init__(
        self,
        in_channels: int,
        stride: tuple[int, int, int],
        *,
        upscale_factor: int = 1,
        causal: bool = False,
        spatial_padding_mode: str = "zeros",
    ):
        super().__init__()
        self.stride = stride
        self.upscale_factor = upscale_factor
        st, sh, sw = stride
        out_channels = (in_channels * st * sh * sw) // upscale_factor
        self.conv = Conv3dBlock(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            causal=causal,
            spatial_padding_mode=spatial_padding_mode,
        )
        self._spatial_factor = sh  # assume sh == sw
        self._temporal_factor = st

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv(x)
        sf = self._spatial_factor
        tf = self._temporal_factor
        if sf > 1 or tf > 1:
            x = pixel_shuffle_3d(x, spatial_factor=max(sf, 1), temporal_factor=max(tf, 1))
            # Diffusers: hidden_states[:, :, stride[0] - 1 :]
            if tf > 1:
                x = x[:, tf - 1 :, :, :, :]
        return x


class PrunaUpBlock3d(nn.Module):
    """Up block with Pruna skip-width ``conv_in`` (in_channels != out * factor)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        num_layers: int,
        upscale_factor: int,
        upsample_type: str,
        upsample_residual: bool = False,
        spatio_temporal_scale: bool = True,
        causal: bool = False,
        spatial_padding_mode: str = "zeros",
        eps: float = 1e-6,
    ):
        super().__init__()
        if upsample_residual:
            raise NotImplementedError("PrunaVAED ships upsample_residual=false")

        pre = out_channels * upscale_factor
        self.conv_in: DiffusersResnet3d | None = None
        if in_channels != pre:
            self.conv_in = DiffusersResnet3d(
                in_channels,
                pre,
                causal=causal,
                spatial_padding_mode=spatial_padding_mode,
                eps=eps,
            )

        self.upsamplers: list[DiffusersUpsampler3d] | None = None
        if spatio_temporal_scale:
            stride = _STRIDE_BY_TYPE[upsample_type]
            self.upsamplers = [
                DiffusersUpsampler3d(
                    pre,
                    stride,
                    upscale_factor=upscale_factor,
                    causal=causal,
                    spatial_padding_mode=spatial_padding_mode,
                )
            ]

        self.resnets = [
            DiffusersResnet3d(
                out_channels,
                out_channels,
                causal=causal,
                spatial_padding_mode=spatial_padding_mode,
                eps=eps,
            )
            for _ in range(num_layers)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        if self.conv_in is not None:
            x = self.conv_in(x)
        if self.upsamplers is not None:
            for up in self.upsamplers:
                x = up(x)
        for block in self.resnets:
            x = block(x)
        return x


class PrunaMidBlock3d(nn.Module):
    def __init__(
        self,
        channels: int,
        num_layers: int,
        *,
        causal: bool = False,
        spatial_padding_mode: str = "zeros",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.resnets = [
            DiffusersResnet3d(
                channels,
                channels,
                causal=causal,
                spatial_padding_mode=spatial_padding_mode,
                eps=eps,
            )
            for _ in range(num_layers)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for block in self.resnets:
            x = block(x)
        return x


class VideoDecoderPruna(nn.Module):
    """Pruned LTX-2.3 VAE decoder matching PrunaVAED + patch_diffusers wiring."""

    def __init__(
        self,
        *,
        latent_channels: int = 128,
        out_channels: int = 3,
        block_out_channels: tuple[int, ...] = _DEFAULT_BLOCK_OUT,
        layers_per_block: tuple[int, ...] = _DEFAULT_LAYERS,
        spatio_temporal_scaling: tuple[bool, ...] = _DEFAULT_SPATIO_TEMPORAL,
        upsample_type: tuple[str, ...] = _DEFAULT_UPSAMPLE_TYPE,
        upsample_factor: tuple[int, ...] = _DEFAULT_UPSAMPLE_FACTOR,
        upsample_residual: tuple[bool, ...] = _DEFAULT_UPSAMPLE_RESIDUAL,
        patch_size: int = 4,
        causal: bool = False,
        spatial_padding_mode: str = "zeros",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.patch_size = patch_size
        self._causal = causal

        # Mirror Pruna patch_diffusers.decoder_init channel tracking
        ch = tuple(reversed(block_out_channels))
        layers = tuple(reversed(layers_per_block))
        scaling = tuple(reversed(spatio_temporal_scaling))
        residual = tuple(reversed(upsample_residual))
        factors = tuple(reversed(upsample_factor))
        # upsample_type: only reverse when length == len(ch) - 1 (stock); Pruna has len==len(ch)
        types = upsample_type
        if len(types) == len(ch) - 1:
            types = tuple(reversed(types))

        width = ch[0]
        self.conv_in = Conv3dBlock(
            latent_channels,
            width,
            kernel_size=3,
            padding=1,
            causal=causal,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.mid_block = PrunaMidBlock3d(
            width,
            layers[0],
            causal=causal,
            spatial_padding_mode=spatial_padding_mode,
            eps=eps,
        )

        self.up_blocks: list[PrunaUpBlock3d] = []
        current = width
        for i in range(len(ch)):
            resnet_w = ch[i] // factors[i]
            self.up_blocks.append(
                PrunaUpBlock3d(
                    in_channels=current,
                    out_channels=resnet_w,
                    num_layers=layers[i + 1],
                    upscale_factor=factors[i],
                    upsample_type=types[i],
                    upsample_residual=residual[i],
                    spatio_temporal_scale=scaling[i],
                    causal=causal,
                    spatial_padding_mode=spatial_padding_mode,
                    eps=eps,
                )
            )
            current = resnet_w

        self.conv_out = Conv3dBlock(
            current,
            out_channels * patch_size * patch_size,
            kernel_size=3,
            padding=1,
            causal=causal,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.per_channel_statistics = PerChannelStatistics(latent_channels)

    @classmethod
    def from_config(cls, config: dict[str, Any] | Path) -> VideoDecoderPruna:
        if isinstance(config, (str, Path)):
            with open(config) as f:
                config = json.load(f)
        assert isinstance(config, dict)

        def _tup(key: str, default: tuple) -> tuple:
            val = config.get(key, default)
            return tuple(val) if val is not None else default

        return cls(
            latent_channels=int(config.get("latent_channels", 128)),
            out_channels=int(config.get("out_channels", 3)),
            block_out_channels=_tup("decoder_block_out_channels", _DEFAULT_BLOCK_OUT),
            layers_per_block=_tup("decoder_layers_per_block", _DEFAULT_LAYERS),
            spatio_temporal_scaling=_tup(
                "decoder_spatio_temporal_scaling", _DEFAULT_SPATIO_TEMPORAL
            ),
            upsample_type=_tup("upsample_type", _DEFAULT_UPSAMPLE_TYPE),
            upsample_factor=_tup("upsample_factor", _DEFAULT_UPSAMPLE_FACTOR),
            upsample_residual=_tup("upsample_residual", _DEFAULT_UPSAMPLE_RESIDUAL),
            patch_size=int(config.get("patch_size", 4)),
            causal=bool(config.get("decoder_causal", False)),
            spatial_padding_mode=str(
                config.get("decoder_spatial_padding_mode", "zeros")
            ),
            eps=float(config.get("resnet_norm_eps", 1e-6)),
        )

    def denormalize_latent(self, latent: mx.array) -> mx.array:
        mean = self.per_channel_statistics.mean.reshape(1, 1, 1, 1, -1)
        std = self.per_channel_statistics.std.reshape(1, 1, 1, 1, -1)
        return latent * std + mean

    def decode(self, latent: mx.array) -> mx.array:
        """Decode latent (B, C, F, H, W) → pixels (B, 3, F, H, W) in [-1, 1]."""
        output_dtype = latent.dtype
        flat_params = mlx.utils.tree_flatten(self.parameters())
        weights_dtype = flat_params[0][1].dtype if flat_params else output_dtype
        if latent.dtype != weights_dtype:
            latent = latent.astype(weights_dtype)

        x = latent.transpose(0, 2, 3, 4, 1)
        x = self.denormalize_latent(x)
        x = self.conv_in(x)
        x = self.mid_block(x)
        for block in self.up_blocks:
            x = block(x)
        x = self.conv_out(nn.silu(pixel_norm(x)))
        x = unpatchify_spatial(x, patch_size=self.patch_size)
        return x.transpose(0, 4, 1, 2, 3).astype(output_dtype)


# Reuse stock streaming / tiling entrypoints (they only call ``self.decode``).
from ltx_core_mlx.model.video_vae.video_vae import VideoDecoder as _StockVideoDecoder

VideoDecoderPruna.tiled_decode = _StockVideoDecoder.tiled_decode  # type: ignore[method-assign]
VideoDecoderPruna.decode_and_stream = _StockVideoDecoder.decode_and_stream  # type: ignore[method-assign]
