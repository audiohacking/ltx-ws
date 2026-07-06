"""BFS V3 face swap — dev + CFG + LTXVAddGuide (Comfy-aligned).

Head-swap LoRA is trained for the **dev** transformer with CFG/STG guidance and
full composite guide via ``LTXVAddGuide`` append + crop. Distilled-only sampling
copies the main-panel performance without identity transfer.

See ``docs/FACESWAP_COMFY_GRAPH.md``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.components.guiders import MultiModalGuiderParams, create_multimodal_guider_factory
from ltx_core_mlx.loader import (
    LTXV_LORA_COMFY_RENAMING_MAP,
    LoraStateDictWithStrength,
    SafetensorsStateDictLoader,
    StateDict,
    apply_loras,
)
from ltx_core_mlx.model.transformer.model import X0Model
from ltx_core_mlx.utils.memory import aggressive_cleanup
from ltx_core_mlx.utils.positions import compute_audio_positions, compute_audio_token_count, compute_video_positions
from ltx_core_mlx.utils.weights import apply_quantization
from ltx_ltxv_add_guide import (
    DEFAULT_GUIDE_CRF,
    build_appended_guide_conditioning,
    crop_guides_from_video_tokens,
    encode_guide_video,
    generation_token_count,
)
from ltx_pipelines_mlx.scheduler import ltx2_schedule
from ltx_pipelines_mlx.ti2vid_one_stage import DEFAULT_CFG_SCALE, TI2VidOneStagePipeline
from ltx_pipelines_mlx.utils.helpers import create_noised_state
from ltx_pipelines_mlx.utils.samplers import guided_denoise_loop

logger = logging.getLogger(__name__)

_mx_eval = getattr(mx, "eval")  # noqa: B009

DEFAULT_FACE_SWAP_NUM_STEPS = 20
DEFAULT_FACE_SWAP_CFG = DEFAULT_CFG_SCALE
DEFAULT_FACE_SWAP_STG = 1.0
DEFAULT_GUIDE_STRENGTH = 1.0

# Backward-compatible aliases
DEFAULT_FACE_SWAP_STAGE1_STEPS = DEFAULT_FACE_SWAP_NUM_STEPS
DEFAULT_FACE_SWAP_STAGE2_STEPS = 0


def _resolve_dev_transformer(model_dir: Path) -> str:
    for name in ("transformer-dev.safetensors", "transformer.safetensors"):
        if (model_dir / name).exists():
            return name
    return "transformer-dev.safetensors"


class FaceSwapPipeline(TI2VidOneStagePipeline):
    """BFS head-swap: dev + CFG + full composite LTXVAddGuide."""

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
        )
        self._head_swap_lora = [(str(p), float(s)) for p, s in lora_paths]
        self._loras_fused = False

    def _fuse_head_swap_lora(self) -> None:
        if self._loras_fused or not self._head_swap_lora:
            return
        assert self.dit is not None

        if self.low_ram_streaming:
            from ltx_core_mlx.loader.block_streaming import BlockLoraSource

            sources: list = list(object.__getattribute__(self.dit, "_lora_sources"))
            for lora_path, strength in self._head_swap_lora:
                sources.append(
                    BlockLoraSource(
                        lora_path,
                        block_prefix="transformer.transformer_blocks.",
                        strength=strength,
                        sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
                    )
                )
                logger.info("Face swap: attached head-swap LoRA stream %s (strength=%s)", lora_path, strength)
            object.__setattr__(self.dit, "_lora_sources", sources)
            self._loras_fused = True
            return

        import mlx.utils

        model_weights = dict(mlx.utils.tree_flatten(self.dit.parameters()))
        model_sd = StateDict(sd=model_weights, size=0, dtype=set())
        loader = SafetensorsStateDictLoader()
        lora_sds = []
        for lora_path, strength in self._head_swap_lora:
            lora_sd = loader.load(lora_path, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)
            lora_sds.append(LoraStateDictWithStrength(state_dict=lora_sd, strength=strength))
            logger.info("Face swap: loaded head-swap LoRA %s (strength=%s)", lora_path, strength)
        fused_sd = apply_loras(model_sd=model_sd, lora_sd_and_strengths=lora_sds)
        apply_quantization(self.dit, fused_sd.sd)
        self.dit.load_weights(list(fused_sd.sd.items()))
        aggressive_cleanup()
        self._loras_fused = True
        logger.info("Face swap: fused head-swap LoRA into dev transformer")

    def load(self) -> None:
        """Load dev DiT (head-swap LoRA fused) + VAE encoder."""
        if self._loaded:
            return
        if self.dit is None:
            self.dit = self._load_dev_transformer()
        self._fuse_head_swap_lora()
        self._load_vae_encoder()
        self._loaded = True

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
        num_steps: int | None = None,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        cfg_scale: float = DEFAULT_FACE_SWAP_CFG,
        stg_scale: float = DEFAULT_FACE_SWAP_STG,
        guide_strength: float = DEFAULT_GUIDE_STRENGTH,
        guide_frame_idx: int = 0,
        guide_crf: int = DEFAULT_GUIDE_CRF,
    ) -> tuple[mx.array, mx.array]:
        if stage2_steps:
            logger.info("Face swap: stage2_steps ignored (single-stage dev+CFG)")

        steps = num_steps or stage1_steps or DEFAULT_FACE_SWAP_NUM_STEPS
        steps = max(8, int(steps))

        f_lat, h_lat, w_lat, gen_tokens = generation_token_count(num_frames, height, width)
        enc_h = h_lat * 32
        enc_w = w_lat * 32

        video_embeds, audio_embeds, neg_video_embeds, neg_audio_embeds = self._encode_text_with_negative(prompt)

        self.load()
        assert self.dit is not None
        assert self.vae_encoder is not None

        encoded = encode_guide_video(
            guide_video_path,
            encode_height=enc_h,
            encode_width=enc_w,
            num_frames=num_frames,
            frame_rate=frame_rate,
            video_encoder=self.vae_encoder,
            video_patchifier=self.video_patchifier,
            frame_idx=guide_frame_idx,
            crf=guide_crf,
        )
        guide_cond = build_appended_guide_conditioning(encoded, strength=guide_strength)

        audio_t = compute_audio_token_count(num_frames, frame_rate=frame_rate)
        audio_shape = (1, audio_t, 128)
        video_positions = compute_video_positions(f_lat, h_lat, w_lat, frame_rate=frame_rate)
        audio_positions = compute_audio_positions(audio_t)

        video_state = create_noised_state(
            base_shape=(1, gen_tokens, 128),
            conditionings=[guide_cond],
            spatial_dims=(f_lat, h_lat, w_lat),
            positions=video_positions,
            seed=seed,
            sigma=1.0,
        )
        audio_state = create_noised_state(
            base_shape=audio_shape,
            conditionings=[],
            spatial_dims=(f_lat, h_lat, w_lat),
            positions=audio_positions,
            seed=seed + 1,
            sigma=1.0,
        )

        sigmas = ltx2_schedule(steps, num_tokens=gen_tokens)

        vgp = MultiModalGuiderParams(
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            rescale_scale=0.7,
            modality_scale=3.0,
            stg_blocks=[28],
        )
        agp = MultiModalGuiderParams(
            cfg_scale=7.0,
            stg_scale=stg_scale,
            rescale_scale=0.7,
            modality_scale=3.0,
            stg_blocks=[28],
        )
        video_factory = create_multimodal_guider_factory(vgp, negative_context=neg_video_embeds)
        audio_factory = create_multimodal_guider_factory(agp, negative_context=neg_audio_embeds)

        logger.info(
            "Face swap: dev+CFG steps=%d cfg=%.1f stg=%.1f add_guide=composite crf=%d "
            "tokens_gen=%d tokens_guide=%d canvas=%dx%d frames=%d",
            steps,
            cfg_scale,
            stg_scale,
            guide_crf,
            gen_tokens,
            int(encoded.tokens.shape[1]),
            width,
            height,
            num_frames,
        )

        x0_model = X0Model(self.dit)
        self._pre_denoise_flush(video_state, audio_state)
        output = guided_denoise_loop(
            model=x0_model,
            video_state=video_state,
            audio_state=audio_state,
            video_text_embeds=video_embeds,
            audio_text_embeds=audio_embeds,
            video_guider_factory=video_factory,
            audio_guider_factory=audio_factory,
            sigmas=sigmas,
        )
        if self.low_memory:
            aggressive_cleanup()

        gen_tokens_out = crop_guides_from_video_tokens(
            output.video_latent,
            num_generation_tokens=gen_tokens,
        )
        video_latent = self.video_patchifier.unpatchify(gen_tokens_out, (f_lat, h_lat, w_lat))
        audio_latent = self.audio_patchifier.unpatchify(output.audio_latent)
        return video_latent, audio_latent

    def generate_and_save(
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
        num_steps: int | None = None,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        cfg_scale: float = DEFAULT_FACE_SWAP_CFG,
        stg_scale: float = DEFAULT_FACE_SWAP_STG,
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
            num_steps=num_steps,
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            guide_strength=guide_strength,
            guide_crf=guide_crf,
        )

        if self.low_memory:
            self.dit = None
            self.prompt_encoder.free()
            self._loaded = False
            self._loras_fused = False
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
    "DEFAULT_FACE_SWAP_NUM_STEPS",
    "DEFAULT_FACE_SWAP_STAGE1_STEPS",
    "DEFAULT_FACE_SWAP_STAGE2_STEPS",
    "DEFAULT_FACE_SWAP_STG",
    "DEFAULT_GUIDE_STRENGTH",
    "FaceSwapPipeline",
]
