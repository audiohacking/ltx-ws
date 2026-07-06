"""Comfy ``LTXVAddGuide`` / ``LTXVCropGuides`` primitives for local BFS face swap.

Implements the append-keyframe contract from ComfyUI ``nodes_lt.LTXVAddGuide``:
  - VAE-encode guide video → patchify tokens
  - Append guide tokens with denoise_mask = 1 - strength (frozen when strength=1)
  - Positions from full latent grid + frame_idx offset (multi-frame guides)
  - After sampling, crop generation tokens (strip appended guides)

See docs/FACESWAP_MLX_PORT.md and ComfyUI ``comfy_extras/nodes_lt.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mlx.core as mx

from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.conditioning.mask_utils import update_attention_mask
from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_core_mlx.utils.ffmpeg import probe_video_info
from ltx_core_mlx.utils.positions import VIDEO_SPATIAL_SCALE, VIDEO_TEMPORAL_SCALE, compute_video_positions
from ltx_core_mlx.utils.video import load_video_frames_normalized

logger = logging.getLogger(__name__)

_mx_eval = getattr(mx, "eval")  # noqa: B009

DEFAULT_GUIDE_CRF = 33


def vae_compatible_frame_count(num_frames: int, source_num_frames: int) -> int:
    """Round down to ``8k+1`` frames for the LTX video VAE."""
    max_frames = min(int(num_frames), int(source_num_frames))
    k = max(1, (max_frames - 1) // 8)
    return 1 + k * 8


def compute_guide_video_positions(
    num_latent_frames: int,
    height: int,
    width: int,
    *,
    frame_rate: float,
    frame_idx: int = 0,
    num_pixel_frames: int | None = None,
) -> mx.array:
    """Pixel-space RoPE positions for a multi-frame guide (Comfy ``add_keyframe_index``).

    Matches upstream ``VideoConditionByKeyframeIndex``: causal fix at ``frame_idx==0``,
    temporal offset by ``frame_idx`` pixel frames, optional single-frame narrowing.
    """
    if frame_idx == 0:
        positions = compute_video_positions(num_latent_frames, height, width, frame_rate=frame_rate)
    else:
        idx = mx.arange(num_latent_frames).astype(mx.float32)
        f_starts = idx * VIDEO_TEMPORAL_SCALE
        f_ends = (idx + 1) * VIDEO_TEMPORAL_SCALE
        f_mids = (f_starts + f_ends) / 2.0 / frame_rate
        h_mids = mx.arange(height).astype(mx.float32) * VIDEO_SPATIAL_SCALE + VIDEO_SPATIAL_SCALE / 2.0
        w_mids = mx.arange(width).astype(mx.float32) * VIDEO_SPATIAL_SCALE + VIDEO_SPATIAL_SCALE / 2.0
        f_grid = mx.repeat(mx.repeat(f_mids[:, None, None], height, axis=1), width, axis=2)
        h_grid = mx.repeat(mx.repeat(h_mids[None, :, None], num_latent_frames, axis=0), width, axis=2)
        w_grid = mx.repeat(mx.repeat(w_mids[None, None, :], num_latent_frames, axis=0), height, axis=1)
        positions = mx.stack([f_grid, h_grid, w_grid], axis=-1).reshape(-1, 3)[None, :, :].astype(mx.float32)

    if frame_idx != 0:
        offset = float(frame_idx) / float(frame_rate)
        positions = positions.at[:, :, 0].add(offset)

    if num_pixel_frames == 1:
        t_start = positions[:, :, 0:1]
        positions = positions.at[:, :, 0].set(t_start[:, :, 0] + 0.0)
        # Narrow temporal extent to one pixel-frame width in seconds.
        # For single-frame guides the upstream sets end = start + 1 (pixel) / fps.
        positions = positions.at[:, :, 0].set(t_start[:, :, 0] + 1.0 / float(frame_rate))

    return positions.astype(mx.float32)


@dataclass(frozen=True)
class EncodedGuideVideo:
    """VAE-encoded composite guide ready for ``LTXVAddGuide`` conditioning."""

    tokens: mx.array
    positions: mx.array
    latent_frames: int
    latent_height: int
    latent_width: int
    pixel_frames: int


def encode_guide_video(
    guide_path: str,
    *,
    encode_height: int,
    encode_width: int,
    num_frames: int,
    frame_rate: float,
    video_encoder,
    video_patchifier,
    frame_idx: int = 0,
    strength: float = 1.0,
) -> EncodedGuideVideo:
    """Load, VAE-encode, and patchify a composite BFS guide video."""
    del strength  # applied when building conditioning item
    info = probe_video_info(guide_path)
    vae_frames = vae_compatible_frame_count(num_frames, info.num_frames)

    video = load_video_frames_normalized(guide_path, encode_height, encode_width, vae_frames)
    video = (video * 2.0 - 1.0).astype(mx.bfloat16)
    encoded = video_encoder.encode(video)
    _mx_eval(encoded)

    ref_f = int(encoded.shape[2])
    ref_h = int(encoded.shape[3])
    ref_w = int(encoded.shape[4])
    tokens, _ = video_patchifier.patchify(encoded)
    positions = compute_guide_video_positions(
        ref_f,
        ref_h,
        ref_w,
        frame_rate=frame_rate,
        frame_idx=frame_idx,
        num_pixel_frames=vae_frames if frame_idx == 0 else None,
    )

    logger.info(
        "LTXVAddGuide encode: guide=%s vae_frames=%d latent=%dx%dx%d tokens=%d frame_idx=%d",
        guide_path,
        vae_frames,
        ref_w,
        ref_h,
        ref_f,
        int(tokens.shape[1]),
        frame_idx,
    )
    return EncodedGuideVideo(
        tokens=tokens,
        positions=positions,
        latent_frames=ref_f,
        latent_height=ref_h,
        latent_width=ref_w,
        pixel_frames=vae_frames,
    )


class VideoConditionByAppendedGuide:
    """Comfy ``LTXVAddGuide.append_keyframe`` — append frozen guide tokens at sequence end."""

    def __init__(
        self,
        guide_tokens: mx.array,
        guide_positions: mx.array,
        *,
        strength: float = 1.0,
    ):
        self.guide_tokens = guide_tokens
        self.guide_positions = guide_positions
        self.strength = float(strength)

    def apply(self, state: LatentState, spatial_dims: tuple[int, int, int]) -> LatentState:
        num_guide = int(self.guide_tokens.shape[1])
        mask_value = 1.0 - self.strength

        new_latent = mx.concatenate([state.latent, self.guide_tokens], axis=1)
        new_clean = mx.concatenate([state.clean_latent, self.guide_tokens], axis=1)
        guide_mask = mx.full((state.denoise_mask.shape[0], num_guide, 1), mask_value, dtype=state.denoise_mask.dtype)
        new_mask = mx.concatenate([state.denoise_mask, guide_mask], axis=1)

        new_positions = state.positions
        if state.positions is not None:
            new_positions = mx.concatenate([state.positions, self.guide_positions], axis=1)

        f_lat, h_lat, w_lat = spatial_dims
        num_noisy = f_lat * h_lat * w_lat
        new_attn_mask = update_attention_mask(
            latent_state=state,
            attention_mask=None,
            num_noisy_tokens=num_noisy,
            num_new_tokens=num_guide,
            batch_size=state.latent.shape[0],
        )

        return LatentState(
            latent=new_latent,
            clean_latent=new_clean,
            denoise_mask=new_mask,
            positions=new_positions,
            attention_mask=new_attn_mask,
        )


def build_appended_guide_conditioning(
    encoded: EncodedGuideVideo,
    *,
    strength: float = 1.0,
) -> VideoConditionByAppendedGuide:
    return VideoConditionByAppendedGuide(
        guide_tokens=encoded.tokens,
        guide_positions=encoded.positions,
        strength=strength,
    )


def crop_guides_from_video_tokens(
    video_tokens: mx.array,
    *,
    num_generation_tokens: int,
) -> mx.array:
    """Comfy ``LTXVCropGuides`` — keep only generation tokens (strip appended guides)."""
    n = int(num_generation_tokens)
    if video_tokens.shape[1] < n:
        raise ValueError(
            f"Cannot crop {n} generation tokens from sequence of length {video_tokens.shape[1]}"
        )
    return video_tokens[:, :n, :]


def generation_token_count(num_frames: int, height: int, width: int) -> tuple[int, int, int, int]:
    """Return ``(F, H, W, token_count)`` for latent dims at pixel height/width."""
    f_lat, h_lat, w_lat = compute_video_latent_shape(num_frames, height, width)
    return f_lat, h_lat, w_lat, f_lat * h_lat * w_lat


__all__ = [
    "DEFAULT_GUIDE_CRF",
    "EncodedGuideVideo",
    "VideoConditionByAppendedGuide",
    "build_appended_guide_conditioning",
    "compute_guide_video_positions",
    "crop_guides_from_video_tokens",
    "encode_guide_video",
    "generation_token_count",
    "vae_compatible_frame_count",
]
