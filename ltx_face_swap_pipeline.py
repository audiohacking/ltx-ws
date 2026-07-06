"""BFS V3 face swap pipeline for MLX.

Comfy BFS V3 uses dev UNet + ``LTXVAddGuideMulti`` + ``LTXVCropGuides``. MLX has no
AddGuideMulti port; this pipeline extends :class:`ICLoraPipeline` like LipDub:

- Composite guide → IC-LoRA ``video_conditioning`` (full sequence)
- Frame-0 composite → ``VideoConditionByLatentIndex`` (AddGuide frame 0)
- **Stage 2 keeps the head-swap LoRA fused** (stock IC-LoRA reloads clean weights)
- **Stage 2 re-appends composite video conditioning** (stock IC-LoRA drops it)

Guide composition and canvas sizing live in ``ltx_face_swap_compose``.
"""

from __future__ import annotations

import logging

import mlx.core as mx

from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
from ltx_pipelines_mlx.ic_lora import ICLoraPipeline
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS, STAGE_2_SIGMAS
from ltx_pipelines_mlx.utils.helpers import create_noised_state
from ltx_pipelines_mlx.utils.samplers import denoise_loop

logger = logging.getLogger(__name__)

_mx_eval = getattr(mx, "eval")  # noqa: B009


class FaceSwapPipeline(ICLoraPipeline):
    """BFS head-swap: IC-LoRA composite guide with LipDub-style stage-2 conditioning."""

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
        """Generate with BFS composite guide; stage 2 retains LoRA + video ref."""
        if not video_conditioning:
            raise ValueError("Face swap requires composite guide video_conditioning")
        if not (0.0 <= conditioning_attention_strength <= 1.0):
            raise ValueError(
                f"conditioning_attention_strength must be in [0.0, 1.0], "
                f"got {conditioning_attention_strength}"
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
        video_shape = (1, F * H_half * W_half, 128)
        audio_T = compute_audio_token_count(num_frames, frame_rate=frame_rate)
        audio_shape = (1, audio_T, 128)

        video_positions_1 = compute_video_positions(F, H_half, W_half, frame_rate=frame_rate)
        audio_positions = compute_audio_positions(audio_T)

        stage_1_conditionings = self._create_conditionings(
            images=images,
            video_conditioning=video_conditioning,
            height=half_h,
            width=half_w,
            num_frames=num_frames,
            frame_rate=frame_rate,
            conditioning_attention_strength=conditioning_attention_strength,
        )

        video_state = create_noised_state(
            base_shape=video_shape,
            conditionings=stage_1_conditionings,
            spatial_dims=(F, H_half, W_half),
            positions=video_positions_1,
            seed=seed,
            sigma=1.0,
            initial_latent=None,
        )
        audio_state = create_noised_state(
            base_shape=audio_shape,
            conditionings=[],
            spatial_dims=(F, H_half, W_half),
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

        gen_tokens = output_1.video_latent[:, : F * H_half * W_half, :]
        video_half = self.video_patchifier.unpatchify(gen_tokens, (F, H_half, W_half))

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

        H_full = H_half * 2
        W_full = W_half * 2
        enc_h_full = H_full * 32
        enc_w_full = W_full * 32

        # Stage 2: keep head-swap LoRA + composite IC ref (LipDub pattern).
        stage_2_conditionings = self._create_conditionings(
            images=images,
            video_conditioning=video_conditioning,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            conditioning_attention_strength=conditioning_attention_strength,
        )

        if self.low_memory:
            self.image_conditioner.free()
            self.upsampler = None

        video_tokens_up, _ = self.video_patchifier.patchify(video_upscaled)
        sigmas_2 = STAGE_2_SIGMAS[: stage2_steps + 1] if stage2_steps else STAGE_2_SIGMAS
        start_sigma = sigmas_2[0]
        video_positions_2 = compute_video_positions(F, H_full, W_full, frame_rate=frame_rate)

        video_state_2 = create_noised_state(
            base_shape=video_tokens_up.shape,
            conditionings=stage_2_conditionings,
            spatial_dims=(F, H_full, W_full),
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
            spatial_dims=(F, H_full, W_full),
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
            output_2.video_latent[:, : F * H_full * W_full, :],
            (F, H_full, W_full),
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


__all__ = ["FaceSwapPipeline"]
