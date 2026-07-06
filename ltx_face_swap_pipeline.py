"""BFS V3 face swap pipeline for MLX.

Comfy BFS V3 VAE-encodes the composite guide and denoises with the head-swap LoRA.
IC-LoRA ``VideoConditionByReferenceLatent`` append would preserve/copy the guide
pixels (including the original face in the main panel) — wrong for face swap.

This pipeline instead:

- VAE-encodes the composite guide as the denoising starting latent (retake-style)
- Applies frame-0 composite image conditioning (AddGuide frame 0)
- Keeps head-swap LoRA fused through stage 2 (LipDub pattern)
- Does **not** append IC-LoRA reference tokens
"""

from __future__ import annotations

import logging

import mlx.core as mx

from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.conditioning.types.latent_cond import LatentState, noise_latent_state
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.utils.ffmpeg import probe_video_info
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
from ltx_core_mlx.utils.video import load_video_frames_normalized
from ltx_pipelines_mlx.ic_lora import ICLoraPipeline
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS, STAGE_2_SIGMAS
from ltx_pipelines_mlx.utils.helpers import create_noised_state, state_with_conditionings
from ltx_pipelines_mlx.utils.samplers import denoise_loop

logger = logging.getLogger(__name__)

_mx_eval = getattr(mx, "eval")  # noqa: B009


def _vae_compatible_frame_count(num_frames: int, source_num_frames: int) -> int:
    max_frames = min(int(num_frames), int(source_num_frames))
    k = max(1, (max_frames - 1) // 8)
    return 1 + k * 8


def _encode_guide_video_tokens(
    guide_path: str,
    *,
    video_encoder,
    video_patchifier,
    height: int,
    width: int,
    num_frames: int,
) -> tuple[mx.array, tuple[int, int, int]]:
    """VAE-encode the BFS composite guide to generation tokens (Comfy VAEEncode path)."""
    info = probe_video_info(guide_path)
    vae_frames = _vae_compatible_frame_count(num_frames, info.num_frames)
    _, H_lat, W_lat = compute_video_latent_shape(num_frames, height, width)
    enc_h = H_lat * 32
    enc_w = W_lat * 32

    video = load_video_frames_normalized(guide_path, enc_h, enc_w, vae_frames)
    video = (video * 2.0 - 1.0).astype(mx.bfloat16)
    encoded = video_encoder.encode(video)
    _mx_eval(encoded)

    tokens, _ = video_patchifier.patchify(encoded)
    F = int(encoded.shape[2])
    H = int(encoded.shape[3])
    W = int(encoded.shape[4])
    return tokens, (F, H, W)


def _frame_image_conditionings(
    images,
    *,
    enc_h: int,
    enc_w: int,
    spatial_dims: tuple[int, int, int],
    video_encoder,
    frame_rate: float,
) -> list:
    """Frame-0 / keyframe anchors only — no IC-LoRA reference append."""
    if not images:
        return []
    from ltx_pipelines_mlx.utils._orchestration import combined_image_conditionings
    from ltx_pipelines_mlx.utils.args import ImageConditioningInput

    normalized = [
        img if isinstance(img, ImageConditioningInput) else ImageConditioningInput(*img) for img in images
    ]
    return combined_image_conditionings(
        normalized,
        enc_h=enc_h,
        enc_w=enc_w,
        spatial_dims=spatial_dims,
        video_encoder=video_encoder,
        frame_rate=frame_rate,
    )


def _noised_video_state_from_guide(
    guide_tokens: mx.array,
    *,
    spatial_dims: tuple[int, int, int],
    positions: mx.array,
    image_conditionings: list,
    seed: int,
    sigma: float = 1.0,
) -> LatentState:
    """Build a fully-denoisable state from encoded guide latents (retake-style)."""
    dtype = mx.bfloat16
    denoise_mask = mx.ones((1, guide_tokens.shape[1], 1), dtype=dtype)
    state = LatentState(
        latent=guide_tokens,
        clean_latent=guide_tokens,
        denoise_mask=denoise_mask,
        positions=positions,
    )
    if image_conditionings:
        state = state_with_conditionings(state, image_conditionings, spatial_dims)
    return noise_latent_state(state, sigma=sigma, seed=seed)


class FaceSwapPipeline(ICLoraPipeline):
    """BFS head-swap: encoded composite init + head-swap LoRA (no IC ref append)."""

    def __init__(
        self,
        model_dir: str,
        lora_paths: list[tuple[str, float]] | None = None,
        gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit",
        low_memory: bool = True,
        low_ram_streaming: bool = False,
    ):
        if not lora_paths or len(lora_paths) != 1:
            raise ValueError("Face swap requires exactly one head-swap LoRA.")
        super().__init__(
            model_dir=model_dir,
            lora_paths=lora_paths,
            gemma_model_id=gemma_model_id,
            low_memory=low_memory,
            low_ram_streaming=low_ram_streaming,
        )

    def generate_face_swap(
        self,
        prompt: str,
        video_conditioning: list[tuple[str, float]],
        height: int,
        width: int,
        num_frames: int,
        *,
        frame_rate: float,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        images: list[tuple[str, int, float]] | None = None,
        conditioning_attention_strength: float = 1.0,
        skip_stage_2: bool = False,
    ) -> tuple[mx.array, mx.array]:
        """Denoise from VAE-encoded composite guide; head-swap LoRA transforms identity."""
        del conditioning_attention_strength  # IC ref append disabled for face swap
        if not video_conditioning:
            raise ValueError("Face swap requires composite guide video_conditioning")

        guide_path = str(video_conditioning[0][0])
        logger.info(
            "Face swap pipeline: VAE-encode guide init from %s (no IC-LoRA ref append)",
            guide_path,
        )

        self._load_text_encoder()
        video_embeds, audio_embeds = self._encode_text(prompt)
        _mx_eval(video_embeds, audio_embeds)
        if self.low_memory:
            self.prompt_encoder.free()
            aggressive_cleanup()

        self.load()
        assert self.dit is not None
        assert self.vae_encoder is not None

        self._fuse_loras()

        half_h, half_w = height // 2, width // 2
        F, H_half, W_half = compute_video_latent_shape(num_frames, half_h, half_w)
        audio_T = compute_audio_token_count(num_frames, frame_rate=frame_rate)
        audio_shape = (1, audio_T, 128)

        guide_tokens_1, spatial_1 = _encode_guide_video_tokens(
            guide_path,
            video_encoder=self.vae_encoder,
            video_patchifier=self.video_patchifier,
            height=half_h,
            width=half_w,
            num_frames=num_frames,
        )
        F1, H1, W1 = spatial_1
        enc_h_half = H1 * 32
        enc_w_half = W1 * 32
        video_positions_1 = compute_video_positions(F1, H1, W1, frame_rate=frame_rate)
        audio_positions = compute_audio_positions(audio_T)

        image_conds_1 = _frame_image_conditionings(
            images,
            enc_h=enc_h_half,
            enc_w=enc_w_half,
            spatial_dims=(F1, H1, W1),
            video_encoder=self.vae_encoder,
            frame_rate=frame_rate,
        )
        video_state = _noised_video_state_from_guide(
            guide_tokens_1,
            spatial_dims=(F1, H1, W1),
            positions=video_positions_1,
            image_conditionings=image_conds_1,
            seed=seed,
            sigma=1.0,
        )
        audio_state = create_noised_state(
            base_shape=audio_shape,
            conditionings=[],
            spatial_dims=(F1, H1, W1),
            positions=audio_positions,
            seed=seed + 1,
            sigma=1.0,
            initial_latent=None,
        )

        sigmas_1 = DISTILLED_SIGMAS[: stage1_steps + 1] if stage1_steps else DISTILLED_SIGMAS
        x0_model = X0Model(self.dit)
        self._pre_denoise_flush(video_state, audio_state)
        output_1 = denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas_1,
        )
        if self.low_memory:
            aggressive_cleanup()

        gen_tokens = output_1.video_latent[:, : F1 * H1 * W1, :]
        video_half = self.video_patchifier.unpatchify(gen_tokens, (F1, H1, W1))

        if skip_stage_2:
            audio_latent = self.audio_patchifier.unpatchify(output_1.audio_latent)
            return video_half, audio_latent

        assert self.upsampler is not None
        video_mlx = video_half.transpose(0, 2, 3, 4, 1)
        video_denorm = self.vae_encoder.denormalize_latent(video_mlx)
        video_denorm = video_denorm.transpose(0, 4, 1, 2, 3)
        video_upscaled = self.upsampler(video_denorm)
        video_up_mlx = video_upscaled.transpose(0, 2, 3, 4, 1)
        video_up_mlx = self.vae_encoder.normalize_latent(video_up_mlx)
        video_upscaled = video_up_mlx.transpose(0, 4, 1, 2, 3)
        _mx_eval(video_upscaled)

        H_full = H1 * 2
        W_full = W1 * 2
        enc_h_full = H_full * 32
        enc_w_full = W_full * 32

        image_conds_2 = _frame_image_conditionings(
            images,
            enc_h=enc_h_full,
            enc_w=enc_w_full,
            spatial_dims=(F1, H_full, W_full),
            video_encoder=self.vae_encoder,
            frame_rate=frame_rate,
        )

        if self.low_memory:
            self.image_conditioner.free()
            self.upsampler = None

        video_tokens_up, _ = self.video_patchifier.patchify(video_upscaled)
        sigmas_2 = STAGE_2_SIGMAS[: stage2_steps + 1] if stage2_steps else STAGE_2_SIGMAS
        start_sigma = sigmas_2[0]
        video_positions_2 = compute_video_positions(F1, H_full, W_full, frame_rate=frame_rate)

        video_state_2 = create_noised_state(
            base_shape=video_tokens_up.shape,
            conditionings=image_conds_2,
            spatial_dims=(F1, H_full, W_full),
            positions=video_positions_2,
            seed=seed + 2,
            sigma=start_sigma,
            initial_latent=video_tokens_up,
            legacy_scalar_blend=True,
        )

        audio_tokens_1 = output_1.audio_latent
        audio_state_2 = create_noised_state(
            base_shape=audio_tokens_1.shape,
            conditionings=[],
            spatial_dims=(F1, H_full, W_full),
            positions=audio_positions,
            seed=seed + 2,
            sigma=start_sigma,
            initial_latent=audio_tokens_1,
        )

        self._pre_denoise_flush(video_state_2, audio_state_2)
        output_2 = denoise_loop(
            model=x0_model,
            video_state=video_state_2,
            audio_state=audio_state_2,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            sigmas=sigmas_2,
        )
        if self.low_memory:
            aggressive_cleanup()

        video_latent = self.video_patchifier.unpatchify(
            output_2.video_latent[:, : F1 * H_full * W_full, :],
            (F1, H_full, W_full),
        )
        audio_latent = self.audio_patchifier.unpatchify(output_2.audio_latent)
        return video_latent, audio_latent

    def generate_and_save(  # type: ignore[override]
        self,
        prompt: str,
        output_path: str,
        video_conditioning: list[tuple[str, float]],
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        *,
        frame_rate: float,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        images: list[tuple[str, int, float]] | None = None,
        conditioning_attention_strength: float = 1.0,
        skip_stage_2: bool = False,
        **_unused,
    ) -> str:
        video_latent, audio_latent = self.generate_face_swap(
            prompt=prompt,
            video_conditioning=video_conditioning,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            seed=seed,
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            images=images,
            conditioning_attention_strength=conditioning_attention_strength,
            skip_stage_2=skip_stage_2,
        )

        if self.low_memory:
            self.dit = None
            self.prompt_encoder.free()
            self.upsampler = None
            self._loaded = False
            aggressive_cleanup()

        self._load_decoders()
        result = self._decode_and_save_video(
            video_latent,
            audio_latent,
            output_path,
            frame_rate=frame_rate,
        )
        if self.low_memory:
            self.audio_decoder = None
            self.vocoder = None
            aggressive_cleanup()
        return result


__all__ = [
    "FaceSwapPipeline",
    "_encode_guide_video_tokens",
    "_noised_video_state_from_guide",
]
