"""BFS V3 face swap — Comfy-aligned LoRA stack + LTXVAddGuide.

Comfy V3 (``docs/FACESWAP_COMFY_GRAPH.md``) loads the **dev** DiT with:

1. distilled-dynamic LoRA @ 1.0 (turns Dev into distilled-like low-step behaviour)
2. head-swap LoRA @ ~0.98–1.0

then samples with the distilled 8-step sigma table at CFG ≈ 1.0.

This pipeline owns fusion via ``_lora_paths`` / ``_head_swap_lora`` so
``_apply_pending_loras`` never reloads a clean DiT and washes the adapters
(CrossView Stage-2 lesson). Single-stage only — no legacy clean Stage 2.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.components.guiders import MultiModalGuiderParams, create_multimodal_guider_factory
from ltx_core_mlx.loader import (
    LTXV_LORA_BLOCK_PREFIX,
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
from ltx_pipelines_mlx.scheduler import DISTILLED_SIGMAS
from ltx_pipelines_mlx.ti2vid_one_stage import TI2VidOneStagePipeline
from ltx_pipelines_mlx.utils.helpers import create_noised_state
from ltx_pipelines_mlx.utils.samplers import guided_denoise_loop

logger = logging.getLogger(__name__)

_mx_eval = getattr(mx, "eval")  # noqa: B009

# Comfy V3 CFGGuider = 1.0 (effectively single-pass at distilled steps).
DEFAULT_FACE_SWAP_CFG = 1.0
DEFAULT_FACE_SWAP_STG = 0.0
DEFAULT_GUIDE_STRENGTH = 1.0


def _resolve_dev_transformer(model_dir: Path) -> str:
    if (model_dir / "transformer-dev.safetensors").exists():
        return "transformer-dev.safetensors"
    raise RuntimeError(
        "Face swap requires transformer-dev.safetensors in the model directory "
        "(use dgrauet/ltx-2.3-mlx or ltx-2.3-mlx-q8 — distilled-only checkpoints are not enough)."
    )


def _count_lora_matches(model_sd: StateDict, lora_sd: StateDict) -> tuple[int, int]:
    """Return ``(matched_A_B_pairs, total_lora_A_tensors)`` after Comfy remap."""
    total = 0
    matched = 0
    for key in lora_sd.sd:
        if not key.endswith(".lora_A.weight"):
            continue
        total += 1
        prefix = key[: -len(".lora_A.weight")]
        key_b = f"{prefix}.lora_B.weight"
        weight_key = f"{prefix}.weight"
        if key_b in lora_sd.sd and weight_key in model_sd.sd:
            matched += 1
    return matched, total


class FaceSwapPipeline(TI2VidOneStagePipeline):
    """BFS head-swap: Comfy V3 LoRA stack + LTXVAddGuide on the dev DiT."""

    def __init__(
        self,
        model_dir: str,
        lora_paths: list[tuple[str, float]] | None = None,
        gemma_model_id: str = "mlx-community/gemma-3-12b-it-4bit",
        low_memory: bool = True,
        low_ram_streaming: bool = False,
    ):
        if not lora_paths:
            raise ValueError("Face swap requires at least one LoRA (head-swap; optional distilled-dynamic stack).")
        head_named = [p for p, _ in lora_paths if "head_swap" in Path(str(p)).name.lower()]
        if not head_named and len(lora_paths) == 1:
            # Single user LoRA is assumed to be the head-swap adapter.
            pass
        elif not head_named:
            raise ValueError(
                "Face swap LoRA stack must include a head-swap adapter "
                f"(got {[Path(str(p)).name for p, _ in lora_paths]})."
            )
        model_path = Path(model_dir)
        super().__init__(
            model_dir,
            gemma_model_id=gemma_model_id,
            low_memory=low_memory,
            low_ram_streaming=low_ram_streaming,
            dev_transformer=_resolve_dev_transformer(model_path),
        )
        # Own LoRAs the same way ICLoraPipeline does: ``_lora_paths`` marks this
        # pipe as fusion-owned so ``_apply_pending_loras`` never reloads a clean DiT.
        self._head_swap_lora = [(str(p), float(s)) for p, s in lora_paths]
        self._lora_paths = list(self._head_swap_lora)
        self._loras_fused = False
        self.pipeline_type = "face_swap"

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
                        block_prefix=LTXV_LORA_BLOCK_PREFIX,
                        strength=strength,
                        sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
                    )
                )
                logger.info(
                    "Face swap: attached LoRA stream %s (strength=%.3f)",
                    Path(lora_path).name,
                    strength,
                )
            object.__setattr__(self.dit, "_lora_sources", sources)
            self._loras_fused = True
            self._lora_paths = list(self._head_swap_lora)
            return

        import mlx.utils

        model_weights = dict(mlx.utils.tree_flatten(self.dit.parameters()))
        model_sd = StateDict(sd=model_weights, size=0, dtype=set())
        loader = SafetensorsStateDictLoader()
        lora_sds = []
        for lora_path, strength in self._head_swap_lora:
            lora_sd = loader.load(lora_path, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)
            matched, total = _count_lora_matches(model_sd, lora_sd)
            logger.info(
                "Face swap: loaded LoRA %s (strength=%.3f) key match %d/%d",
                Path(lora_path).name,
                strength,
                matched,
                total,
            )
            if total > 0 and matched == 0:
                raise RuntimeError(
                    f"Face swap LoRA incompatible with loaded DiT (0/{total} keys matched): {lora_path}"
                )
            if matched == 0:
                logger.warning(
                    "Face swap: LoRA %s produced no A/B pairs after remap — fusion may be a no-op",
                    Path(lora_path).name,
                )
            lora_sds.append(LoraStateDictWithStrength(state_dict=lora_sd, strength=strength))
        fused_sd = apply_loras(model_sd=model_sd, lora_sd_and_strengths=lora_sds)
        apply_quantization(self.dit, fused_sd.sd)
        self.dit.load_weights(list(fused_sd.sd.items()))
        aggressive_cleanup()
        self._loras_fused = True
        self._lora_paths = list(self._head_swap_lora)
        logger.info(
            "Face swap: fused %d LoRA(s) into dev transformer (owned; no Stage-2 reload)",
            len(self._head_swap_lora),
        )

    def load(self) -> None:
        """Load dev DiT (LoRA stack fused) + VAE encoder."""
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
            logger.info("Face swap: stage2_steps ignored (single-stage Comfy V3; LoRAs stay fused)")

        if not (num_steps or stage1_steps):
            raise ValueError("face_swap requires num_steps from the generation request")
        steps = int(num_steps or stage1_steps)
        if steps < 1:
            raise ValueError(f"face_swap num_steps must be >= 1, got {steps}")
        max_distilled_steps = len(DISTILLED_SIGMAS) - 1
        if steps > max_distilled_steps:
            logger.info(
                "Face swap: clamping steps %d → %d (DISTILLED_SIGMAS)",
                steps,
                max_distilled_steps,
            )
            steps = max_distilled_steps

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

        # Comfy V3 / IC distilled path: fixed 8-step table (not token-shifted ltx2_schedule).
        sigmas = DISTILLED_SIGMAS[: steps + 1]

        vgp = MultiModalGuiderParams(
            cfg_scale=cfg_scale,
            stg_scale=stg_scale,
            rescale_scale=0.7,
            modality_scale=3.0,
            stg_blocks=[28],
        )
        agp = MultiModalGuiderParams(
            cfg_scale=max(cfg_scale, 1.0) if cfg_scale > 1.0 else 1.0,
            stg_scale=stg_scale,
            rescale_scale=0.7,
            modality_scale=3.0,
            stg_blocks=[28],
        )
        video_factory = create_multimodal_guider_factory(vgp, negative_context=neg_video_embeds)
        audio_factory = create_multimodal_guider_factory(agp, negative_context=neg_audio_embeds)

        logger.info(
            "Face swap: steps=%d cfg=%.1f stg=%.1f schedule=DISTILLED_SIGMAS "
            "loras=%d add_guide=composite crf=%d tokens_gen=%d tokens_guide=%d "
            "canvas=%dx%d frames=%d",
            steps,
            cfg_scale,
            stg_scale,
            len(self._head_swap_lora),
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
    "DEFAULT_FACE_SWAP_STG",
    "FaceSwapPipeline",
    "_count_lora_matches",
]
