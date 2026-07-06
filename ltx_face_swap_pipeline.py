"""BFS V3 face swap on MLX via dev transformer + CFG + composite keyframes.

Comfy BFS V3 uses dev UNet, CFG, composite guide, and ``LTXVAddGuideMulti``.
Distilled IC-LoRA reference-append copies the guide (original face + motion) or,
without it, loses motion entirely. Keyframe interpolation on the **dev** model
with composite keyframes (interval 8) matches the persistent side-panel identity
signal and main-panel motion from the BFS composite guide.
"""

from __future__ import annotations

import logging
from pathlib import Path

import mlx.core as mx

from ltx_core_mlx.components.guiders import MultiModalGuiderParams
from ltx_pipelines_mlx.ic_lora import ICLoraPipeline
from ltx_pipelines_mlx.keyframe_interpolation import KeyframeInterpolationPipeline

logger = logging.getLogger(__name__)

DEFAULT_FACE_SWAP_CFG = 3.0
DEFAULT_FACE_SWAP_STAGE1_STEPS = 20
DEFAULT_FACE_SWAP_STAGE2_STEPS = 3


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
    """BFS head-swap: dev + CFG + composite keyframes + head-swap LoRA."""

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
        dev_name = _resolve_dev_transformer(model_path)
        lora_name = _resolve_distilled_lora(model_path)
        super().__init__(
            model_dir,
            gemma_model_id=gemma_model_id,
            low_memory=low_memory,
            low_ram_streaming=low_ram_streaming,
            dev_transformer=dev_name,
            distilled_lora=lora_name,
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
        keyframe_tmpdir: str | Path,
        height: int,
        width: int,
        num_frames: int,
        *,
        frame_rate: float,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        cfg_scale: float = DEFAULT_FACE_SWAP_CFG,
        keyframe_interval: int = 8,
    ) -> tuple[mx.array, mx.array]:
        from ltx_face_swap_compose import (
            DEFAULT_BFS_GUIDE_KEYFRAME_INTERVAL,
            extract_bfs_guide_keyframe_images,
        )

        interval = max(1, int(keyframe_interval or DEFAULT_BFS_GUIDE_KEYFRAME_INTERVAL))
        keyframe_tmpdir = Path(keyframe_tmpdir)
        keyframe_tmpdir.mkdir(parents=True, exist_ok=True)

        keyframes = extract_bfs_guide_keyframe_images(
            guide_video_path,
            keyframe_tmpdir,
            num_frames=num_frames,
            interval=interval,
        )
        kf_images = [path for path, *_ in keyframes]
        kf_indices = [idx for _, idx, *_ in keyframes]
        kf_strengths = [float(strength) for _, _, strength, *_ in keyframes]

        logger.info(
            "Face swap pipeline: dev+CFG keyframes=%d interval=%d guide=%s",
            len(kf_images),
            interval,
            guide_video_path,
        )

        if self.dit is None:
            self.dit = self._load_dev_transformer()
        self._fuse_head_swap_loras()

        return self.interpolate(
            prompt,
            kf_images,
            kf_indices,
            kf_strengths,
            height,
            width,
            num_frames,
            frame_rate=frame_rate,
            seed=seed,
            stage1_steps=stage1_steps or DEFAULT_FACE_SWAP_STAGE1_STEPS,
            stage2_steps=stage2_steps or DEFAULT_FACE_SWAP_STAGE2_STEPS,
            cfg_scale=cfg_scale,
            video_guider_params=MultiModalGuiderParams(
                cfg_scale=cfg_scale,
                stg_scale=0.0,
                rescale_scale=0.7,
                modality_scale=3.0,
                stg_blocks=[28],
            ),
            audio_guider_params=MultiModalGuiderParams(
                cfg_scale=7.0,
                stg_scale=0.0,
                rescale_scale=0.7,
                modality_scale=3.0,
                stg_blocks=[28],
            ),
        )

    def generate_and_save(  # type: ignore[override]
        self,
        prompt: str,
        output_path: str,
        guide_video_path: str,
        keyframe_tmpdir: str | Path,
        height: int = 480,
        width: int = 704,
        num_frames: int = 97,
        *,
        frame_rate: float,
        seed: int = 42,
        stage1_steps: int | None = None,
        stage2_steps: int | None = None,
        cfg_scale: float = DEFAULT_FACE_SWAP_CFG,
        **_unused,
    ) -> str:
        video_latent, audio_latent = self.generate_face_swap(
            prompt=prompt,
            guide_video_path=guide_video_path,
            keyframe_tmpdir=keyframe_tmpdir,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            seed=seed,
            stage1_steps=stage1_steps,
            stage2_steps=stage2_steps,
            cfg_scale=cfg_scale,
        )

        if self.low_memory:
            self.dit = None
            self.prompt_encoder.free()
            self.upsampler = None
            self._loaded = False
            self._head_swap_fused = False
            from ltx_core_mlx.utils.memory import aggressive_cleanup

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
            from ltx_core_mlx.utils.memory import aggressive_cleanup

            aggressive_cleanup()
        return result


__all__ = [
    "DEFAULT_FACE_SWAP_CFG",
    "DEFAULT_FACE_SWAP_STAGE1_STEPS",
    "DEFAULT_FACE_SWAP_STAGE2_STEPS",
    "FaceSwapPipeline",
]
