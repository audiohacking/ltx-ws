"""BFS V3 face swap — Comfy ``LTXVAddGuide`` + dev CFG + head-swap LoRA.

Uses local :mod:`ltx_ltxv_add_guide` (append-keyframe + crop-guides) with the full
composite BFS guide video at ``frame_idx=0``. See docs/FACESWAP_MLX_PORT.md.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.components.guiders import MultiModalGuiderParams, create_multimodal_guider_factory
from ltx_core_mlx.components.patchifiers import compute_video_latent_shape
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
from ltx_ltxv_add_guide import (
    DEFAULT_GUIDE_CRF,
    build_appended_guide_conditioning,
    crop_guides_from_video_tokens,
    encode_guide_video,
    generation_token_count,
)
from ltx_pipelines_mlx.ic_lora import ICLoraPipeline
from ltx_pipelines_mlx.keyframe_interpolation import KeyframeInterpolationPipeline
from ltx_pipelines_mlx.scheduler import STAGE_2_SIGMAS, ltx2_schedule
from ltx_pipelines_mlx.utils.helpers import create_noised_state
from ltx_pipelines_mlx.utils.samplers import denoise_loop, guided_denoise_loop

logger = logging.getLogger(__name__)

DEFAULT_FACE_SWAP_CFG = 3.0
DEFAULT_FACE_SWAP_STAGE1_STEPS = 20
DEFAULT_FACE_SWAP_STAGE2_STEPS = 3
DEFAULT_GUIDE_STRENGTH = 1.0


def _pick_model_file(model_dir: Path, *candidates: str) -> str:
    for name in candidates:
        if (model_dir / name).exists():
            return name
    return candidates[0]


def _resolve_dev_transformer(model_dir: Path) -> str:
    return _pick_model_file(
        model_dir,
        "transformer.safetensors",
        "transformer-dev.safetensors",
    )


def _resolve_distilled_lora(model_dir: Path) -> str:
    return _pick_model_file(
        model_dir,
        "ltx-2.3-22b-distilled-lora-384.safetensors",
        "ltx-2.3-22b-distilled-lora.safetensors",
    )


class FaceSwapPipeline(KeyframeInterpolationPipeline):
    """BFS head-swap: full composite guide via LTXVAddGuide + dev CFG + LoRA."""

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
        model_path = Path(model_dir)
        super().__init__(
            model_dir,
            gemma_model_id=gemma_model_id,
            low_memory=low_memory,
            low_ram_streaming=low_ram_streaming,
            dev_transformer=_resolve_dev_transformer(model_path),
            distilled_lora=_resolve_distilled_lora(model_path),
        )
        self._lora_paths = [(str(p), float(s)) for p, s in lora_paths]
        self._head_swap_fused = False

    def _fuse_head_swap_loras(self) -> None:
        if self._head_swap_fused or not self._lora_paths:
            return
        assert self.dit is not None
        ICLoraPipeline._fuse_loras(self)
        self._head_swap_fused = True
        logger.info("Face swap: fused head-swap LoRA into dev transformer")

    def generate_face_swap(
        self,
        prompt: str,
        guide_video_path: str,
        height: int,
        width: int,
        num_frames: int,
        *,
        frame_rate: float,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        cfg_scale: float = DEFAULT_FACE_SWAP_CFG,
        guide_strength: float = DEFAULT_GUIDE_STRENGTH,
        guide_frame_idx: int = 0,
        guide_crf: int = DEFAULT_GUIDE_CRF,
    ) -> tuple[mx.array, mx.array]:
        half_h, half_w = height // 2, width // 2
        f_half, h_half, w_half = compute_video_latent_shape(num_frames, half_h, half_w)
        enc_h_half = h_half * 32
        enc_w_half = w_half * 32
        _, _, _, gen_tokens_half = generation_token_count(num_frames, half_h, half_w)

        h_full, w_full = height, width
        f_full, h_lat_full, w_lat_full, gen_tokens_full = generation_token_count(num_frames, h_full, w_full)
        enc_h_full = h_lat_full * 32
        enc_w_full = w_lat_full * 32

        self._load_vae_encoder()
        assert self.vae_encoder is not None

        encoded_half = encode_guide_video(
            guide_video_path,
            encode_height=enc_h_half,
            encode_width=enc_w_half,
            num_frames=num_frames,
            frame_rate=frame_rate,
            video_encoder=self.vae_encoder,
            video_patchifier=self.video_patchifier,
            frame_idx=guide_frame_idx,
            crf=guide_crf,
        )
        guide_cond_half = build_appended_guide_conditioning(encoded_half, strength=guide_strength)

        video_embeds, audio_embeds, neg_video_embeds, neg_audio_embeds = self._encode_text_with_negative(prompt)

        if self.dit is None:
            self.dit = self._load_dev_transformer()
        self._fuse_head_swap_loras()
        if self.upsampler is None:
            self._load_upsampler()
        assert self.dit is not None
        assert self.upsampler is not None

        audio_t = compute_audio_token_count(num_frames, frame_rate=frame_rate)
        audio_shape = (1, audio_t, 128)
        video_positions_half = compute_video_positions(f_half, h_half, w_half, frame_rate=frame_rate)
        audio_positions = compute_audio_positions(audio_t)

        video_state_1 = create_noised_state(
            base_shape=(1, gen_tokens_half, 128),
            conditionings=[guide_cond_half],
            spatial_dims=(f_half, h_half, w_half),
            positions=video_positions_half,
            seed=seed,
            sigma=1.0,
        )
        audio_state_1 = create_noised_state(
            base_shape=audio_shape,
            conditionings=[],
            spatial_dims=(f_half, h_half, w_half),
            positions=audio_positions,
            seed=seed + 1,
            sigma=1.0,
        )

        s1_steps = stage1_steps or DEFAULT_FACE_SWAP_STAGE1_STEPS
        sigmas_1 = ltx2_schedule(s1_steps, num_tokens=gen_tokens_half)
        x0_model = X0Model(self.dit)

        vgp = MultiModalGuiderParams(
            cfg_scale=cfg_scale,
            stg_scale=0.0,
            rescale_scale=0.7,
            modality_scale=3.0,
            stg_blocks=[28],
        )
        agp = MultiModalGuiderParams(
            cfg_scale=7.0,
            stg_scale=0.0,
            rescale_scale=0.7,
            modality_scale=3.0,
            stg_blocks=[28],
        )
        video_factory = create_multimodal_guider_factory(vgp, negative_context=neg_video_embeds)
        audio_factory = create_multimodal_guider_factory(agp, negative_context=neg_audio_embeds)

        logger.info(
            "Face swap stage1: dev+CFG steps=%d cfg=%.1f add_guide=full_composite crf=%d "
            "tokens_gen=%d tokens_guide=%d",
            s1_steps,
            cfg_scale,
            guide_crf,
            gen_tokens_half,
            int(encoded_half.tokens.shape[1]),
        )

        self._pre_denoise_flush(video_state_1, audio_state_1)
        output_1 = guided_denoise_loop(
            model=x0_model,
            video_state=video_state_1,
            audio_state=audio_state_1,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            video_guider_factory=video_factory,
            audio_guider_factory=audio_factory,
            sigmas=sigmas_1,
        )
        if self.low_memory:
            aggressive_cleanup()

        gen_tokens_1 = crop_guides_from_video_tokens(output_1.video_latent, num_generation_tokens=gen_tokens_half)

        if self._distilled_lora:
            self._fuse_distilled_lora(self.dit)

        video_half = self.video_patchifier.unpatchify(gen_tokens_1, (f_half, h_half, w_half))

        def _denorm_upscale_renorm(encoder) -> mx.array:
            v_mlx = video_half.transpose(0, 2, 3, 4, 1)
            v_denorm = encoder.denormalize_latent(v_mlx).transpose(0, 4, 1, 2, 3)
            v_up = self.upsampler(v_denorm)
            v_up_mlx = encoder.normalize_latent(v_up.transpose(0, 2, 3, 4, 1))
            return v_up_mlx.transpose(0, 4, 1, 2, 3)

        video_upscaled = self.image_conditioner(_denorm_upscale_renorm, free_after=True)
        mx.async_eval(video_upscaled)
        if self.low_memory:
            self.upsampler = None
            aggressive_cleanup()

        encoded_full = encode_guide_video(
            guide_video_path,
            encode_height=enc_h_full,
            encode_width=enc_w_full,
            num_frames=num_frames,
            frame_rate=frame_rate,
            video_encoder=self.vae_encoder,
            video_patchifier=self.video_patchifier,
            frame_idx=guide_frame_idx,
            crf=guide_crf,
        )
        guide_cond_full = build_appended_guide_conditioning(encoded_full, strength=guide_strength)

        video_tokens_up, _ = self.video_patchifier.patchify(video_upscaled)
        sigmas_2 = STAGE_2_SIGMAS[: (stage2_steps or DEFAULT_FACE_SWAP_STAGE2_STEPS) + 1]
        start_sigma = sigmas_2[0]
        video_positions_full = compute_video_positions(f_full, h_lat_full, w_lat_full, frame_rate=frame_rate)

        video_state_2 = create_noised_state(
            base_shape=video_tokens_up.shape,
            conditionings=[guide_cond_full],
            spatial_dims=(f_full, h_lat_full, w_lat_full),
            positions=video_positions_full,
            seed=seed + 2,
            sigma=start_sigma,
            initial_latent=video_tokens_up,
        )
        audio_state_2 = create_noised_state(
            base_shape=output_1.audio_latent.shape,
            conditionings=[],
            spatial_dims=(f_full, h_lat_full, w_lat_full),
            positions=audio_positions,
            seed=seed + 2,
            sigma=start_sigma,
            initial_latent=output_1.audio_latent,
        )

        logger.info(
            "Face swap stage2: distilled refine steps=%d crop_guides_after=yes",
            len(sigmas_2) - 1,
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

        gen_tokens_2 = crop_guides_from_video_tokens(output_2.video_latent, num_generation_tokens=gen_tokens_full)
        video_latent = self.video_patchifier.unpatchify(gen_tokens_2, (f_full, h_lat_full, w_lat_full))
        audio_latent = self.audio_patchifier.unpatchify(output_2.audio_latent)
        return video_latent, audio_latent

    def generate_and_save(  # type: ignore[override]
        self,
        prompt: str,
        output_path: str,
        guide_video_path: str,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        *,
        frame_rate: float,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        cfg_scale: float = DEFAULT_FACE_SWAP_CFG,
        guide_strength: float = DEFAULT_GUIDE_STRENGTH,
        guide_crf: int = DEFAULT_GUIDE_CRF,
        **_unused,
    ) -> str:
        video_latent, audio_latent = self.generate_face_swap(
            prompt=prompt,
            guide_video_path=guide_video_path,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            seed=seed,
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            cfg_scale=cfg_scale,
            guide_strength=guide_strength,
            guide_crf=guide_crf,
        )

        if self.low_memory:
            self.dit = None
            self.prompt_encoder.free()
            self.upsampler = None
            self._loaded = False
            self._head_swap_fused = False
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
    "DEFAULT_FACE_SWAP_CFG",
    "DEFAULT_FACE_SWAP_STAGE1_STEPS",
    "DEFAULT_FACE_SWAP_STAGE2_STEPS",
    "DEFAULT_GUIDE_STRENGTH",
    "FaceSwapPipeline",
]
