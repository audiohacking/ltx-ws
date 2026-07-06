"""Comfy ``LTXVAddGuide`` / ``LTXVCropGuides`` / ``LTXVPreprocess`` for BFS face swap.

Local port of ComfyUI ``comfy_extras/nodes_lt.py`` guide path:
  - ``LTXVPreprocess`` (H.264 CRF round-trip per frame)
  - VAE encode + ``append_keyframe`` (zeros in noisy latent, clean guide tokens)
  - ``LTXVCropGuides`` after sampling

See docs/FACESWAP_MLX_PORT.md.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import mlx.core as mx
import numpy as np

from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.conditioning.mask_utils import update_attention_mask
from ltx_core_mlx.conditioning.types.latent_cond import LatentState
from ltx_core_mlx.utils.ffmpeg import probe_video_info
from ltx_core_mlx.utils.positions import (
    VIDEO_SPATIAL_SCALE,
    VIDEO_TEMPORAL_SCALE,
)
from ltx_core_mlx.utils.video import load_video_frames_normalized

logger = logging.getLogger(__name__)

_mx_eval = getattr(mx, "eval")  # noqa: B009

DEFAULT_GUIDE_CRF = 33
DEFAULT_GUIDE_BLUR_RADIUS = 0
VIDEO_TIME_SCALE = 8


def vae_compatible_frame_count(num_frames: int, source_num_frames: int | None = None) -> int:
    """Round down to ``8k+1`` pixel frames (LTX video VAE temporal layout)."""
    max_frames = int(num_frames)
    if source_num_frames is not None:
        max_frames = min(max_frames, int(source_num_frames))
    k = max(1, (max_frames - 1) // VIDEO_TIME_SCALE)
    return 1 + k * VIDEO_TIME_SCALE


def ltxv_preprocess_rgb_frame(frame: np.ndarray, *, crf: int = DEFAULT_GUIDE_CRF) -> np.ndarray:
    """Comfy ``LTXVPreprocess`` — single RGB frame float ``[0,1]`` HWC in/out."""
    if crf <= 0:
        return frame
    import av

    h, w = frame.shape[:2]
    h2, w2 = (h // 2) * 2, (w // 2) * 2
    rgb = (np.clip(frame[:h2, :w2], 0.0, 1.0) * 255.0).astype(np.uint8)

    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="mp4")
    try:
        stream = container.add_stream("libx264", rate=1, options={"crf": str(int(crf)), "preset": "veryfast"})
        stream.height = rgb.shape[0]
        stream.width = rgb.shape[1]
        av_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(format="yuv420p")
        container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
    finally:
        container.close()

    decoded = io.BytesIO(buf.getvalue())
    with av.open(decoded) as dec:
        vstream = next(s for s in dec.streams if s.type == "video")
        out = next(dec.decode(vstream)).to_ndarray(format="rgb24")
    return out.astype(np.float32) / 255.0


def _mlx_to_numpy_f32(video: mx.array) -> np.ndarray:
    """Materialize MLX video tensor as float32 numpy (bfloat16 is not PEP-3118 safe)."""
    _mx_eval(video)
    return np.asarray(video.astype(mx.float32))


def preprocess_guide_video_tensor(
    video: mx.array,
    *,
    crf: int = DEFAULT_GUIDE_CRF,
    blur_radius: int = DEFAULT_GUIDE_BLUR_RADIUS,
) -> mx.array:
    """Apply ``LTXVPreprocess`` (+ optional blur) to ``(1, 3, F, H, W)`` in ``[0,1]``."""
    del blur_radius  # Comfy blur optional; BFS workflows use 0
    if crf <= 0:
        return video.astype(mx.float32)
    arr = _mlx_to_numpy_f32(video)
    if arr.ndim != 5:
        raise ValueError(f"Expected video (1, 3, F, H, W), got {arr.shape}")
    _, _, f, _, _ = arr.shape
    out_frames = []
    for fi in range(f):
        hwc = np.transpose(arr[0, :, fi], (1, 2, 0))
        out_frames.append(ltxv_preprocess_rgb_frame(hwc, crf=crf))
    stacked = np.stack(out_frames, axis=0)  # F,H,W,C
    stacked = np.transpose(stacked, (3, 0, 1, 2))[None, ...]  # 1,C,F,H,W
    return mx.array(stacked, dtype=mx.float32)


def _video_patch_grid_bounds_np(
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    *,
    patch_size: tuple[int, int, int] = (1, 1, 1),
) -> np.ndarray:
    """Comfy / ltx-core ``get_patch_grid_bounds`` for video latents (batch=1)."""
    pt, ph, pw = patch_size
    f_grid = np.arange(0, num_latent_frames, pt, dtype=np.float32)
    h_grid = np.arange(0, latent_height, ph, dtype=np.float32)
    w_grid = np.arange(0, latent_width, pw, dtype=np.float32)
    f_starts, h_starts, w_starts = np.meshgrid(f_grid, h_grid, w_grid, indexing="ij")
    f_ends = f_starts + pt
    h_ends = h_starts + ph
    w_ends = w_starts + pw
    starts = np.stack([f_starts, h_starts, w_starts], axis=0)
    ends = np.stack([f_ends, h_ends, w_ends], axis=0)
    bounds = np.stack([starts, ends], axis=-1)  # (3, F, H, W, 2)
    flat = bounds.reshape(3, -1, 2)
    return flat[np.newaxis, ...]  # (1, 3, N, 2)


def _pixel_coords_from_bounds_np(
    latent_bounds: np.ndarray,
    *,
    causal_fix: bool,
) -> np.ndarray:
    """Comfy ``latent_to_pixel_coords`` / ltx-core ``get_pixel_coords``."""
    scale = np.array(
        [VIDEO_TEMPORAL_SCALE, VIDEO_SPATIAL_SCALE, VIDEO_SPATIAL_SCALE],
        dtype=np.float32,
    ).reshape(1, 3, 1, 1)
    pixel = latent_bounds.astype(np.float32) * scale
    if causal_fix:
        pixel[:, 0, ...] = np.maximum(pixel[:, 0, ...] + 1.0 - VIDEO_TEMPORAL_SCALE, 0.0)
    return pixel


def _rope_positions_from_pixel_bounds(
    pixel_bounds: np.ndarray,
    *,
    frame_rate: float,
) -> np.ndarray:
    """Token RoPE positions: temporal mid / fps, spatial mids in pixels."""
    t_mid = (pixel_bounds[:, 0, :, 0] + pixel_bounds[:, 0, :, 1]) * 0.5 / float(frame_rate)
    h_mid = (pixel_bounds[:, 1, :, 0] + pixel_bounds[:, 1, :, 1]) * 0.5
    w_mid = (pixel_bounds[:, 2, :, 0] + pixel_bounds[:, 2, :, 1]) * 0.5
    return np.stack([t_mid, h_mid, w_mid], axis=-1).astype(np.float32)


def compute_guide_video_positions(
    num_latent_frames: int,
    height: int,
    width: int,
    *,
    frame_rate: float,
    frame_idx: int = 0,
    num_pixel_frames: int | None = None,
) -> mx.array:
    """RoPE positions for appended guide tokens (Comfy ``add_keyframe_index``)."""
    del num_pixel_frames
    causal_fix = frame_idx == 0 or num_latent_frames == 1
    bounds = _video_patch_grid_bounds_np(num_latent_frames, height, width)
    pixel = _pixel_coords_from_bounds_np(bounds, causal_fix=causal_fix)
    if frame_idx != 0:
        pixel = pixel.copy()
        pixel[:, 0, :, :] += float(frame_idx)
    rope = _rope_positions_from_pixel_bounds(pixel, frame_rate=frame_rate)
    return mx.array(rope, dtype=mx.float32)


@dataclass(frozen=True)
class EncodedGuideVideo:
    """VAE-encoded composite guide for ``LTXVAddGuide``."""

    tokens: mx.array
    positions: mx.array
    latent_frames: int
    latent_height: int
    latent_width: int
    pixel_frames: int
    num_guide_tokens: int


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
    crf: int = DEFAULT_GUIDE_CRF,
    blur_radius: int = DEFAULT_GUIDE_BLUR_RADIUS,
) -> EncodedGuideVideo:
    """Load, preprocess, VAE-encode, and patchify a BFS composite guide video."""
    info = probe_video_info(guide_path)
    vae_frames = vae_compatible_frame_count(num_frames, info.num_frames)

    video = load_video_frames_normalized(guide_path, encode_height, encode_width, vae_frames)
    video = preprocess_guide_video_tensor(video, crf=crf, blur_radius=blur_radius)
    video = (video * 2.0 - 1.0).astype(mx.bfloat16)

    # Comfy ``LTXVAddGuide.encode``: trim to VAE-compatible length before encode.
    _, _, f_in, _, _ = video.shape
    keep = (f_in - 1) // VIDEO_TIME_SCALE * VIDEO_TIME_SCALE + 1
    if keep < f_in:
        video = video[:, :, :keep, :, :]

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
        num_pixel_frames=None,
    )
    n_guide = int(tokens.shape[1])

    logger.info(
        "LTXVAddGuide: guide=%s crf=%d vae_frames=%d latent=%dx%dx%d "
        "tokens_gen_guide=%d frame_idx=%d",
        guide_path,
        crf,
        vae_frames,
        ref_w,
        ref_h,
        ref_f,
        n_guide,
        frame_idx,
    )
    return EncodedGuideVideo(
        tokens=tokens,
        positions=positions,
        latent_frames=ref_f,
        latent_height=ref_h,
        latent_width=ref_w,
        pixel_frames=vae_frames,
        num_guide_tokens=n_guide,
    )


class VideoConditionByAppendedGuide:
    """Comfy ``LTXVAddGuide.append_keyframe`` — append clean guide; noisy slots are zeros."""

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
        guide_zeros = mx.zeros_like(self.guide_tokens)

        new_latent = mx.concatenate([state.latent, guide_zeros], axis=1)
        new_clean = mx.concatenate([state.clean_latent, self.guide_tokens], axis=1)
        guide_mask = mx.full(
            (state.denoise_mask.shape[0], num_guide, 1),
            mask_value,
            dtype=state.denoise_mask.dtype,
        )
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
    """``LTXVCropGuides`` — generation tokens only."""
    n = int(num_generation_tokens)
    if video_tokens.shape[1] < n:
        raise ValueError(
            f"Cannot crop {n} generation tokens from sequence length {video_tokens.shape[1]}"
        )
    return video_tokens[:, :n, :]


def generation_token_count(num_frames: int, height: int, width: int) -> tuple[int, int, int, int]:
    f_lat, h_lat, w_lat = compute_video_latent_shape(num_frames, height, width)
    return f_lat, h_lat, w_lat, f_lat * h_lat * w_lat


__all__ = [
    "DEFAULT_GUIDE_BLUR_RADIUS",
    "DEFAULT_GUIDE_CRF",
    "EncodedGuideVideo",
    "VideoConditionByAppendedGuide",
    "build_appended_guide_conditioning",
    "compute_guide_video_positions",
    "crop_guides_from_video_tokens",
    "encode_guide_video",
    "generation_token_count",
    "ltxv_preprocess_rgb_frame",
    "preprocess_guide_video_tensor",
    "vae_compatible_frame_count",
]
