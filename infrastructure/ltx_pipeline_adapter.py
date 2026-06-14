# -*- coding: utf-8 -*-
"""
Adaptador de infraestructura para pipelines LTX Video 2.3.
Encapsula ltx_backend.py y traduce entre domain models y la API interna de LTX.
"""
import logging
import os
import tempfile
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image

try:
    from domain.models import LTXGenerationSettings, LTXPipelineConfig
except ImportError:
    from ..domain.models import LTXGenerationSettings, LTXPipelineConfig

try:
    from ltx_backend import (
        LTXDistilledGGUFVideoPipeline,
        LTXFastVideoPipeline,
        default_tiling_config,
        extract_video_audio,
    )
except ImportError:
    from ..ltx_backend import (
        LTXDistilledGGUFVideoPipeline,
        LTXFastVideoPipeline,
        default_tiling_config,
        extract_video_audio,
    )

logger = logging.getLogger(__name__)


class LTXPipelineAdapter:
    """
    Infrastructure adapter that wraps LTX 2.3 video generation pipelines.
    Responsible for model loading, inference execution, and tensor normalization.
    """

    @staticmethod
    def load_pipeline(config: LTXPipelineConfig) -> Any:
        """
        Carga el pipeline LTX apropiado (Fast o Distilled GGUF) según la configuración.
        """
        torch_device = torch.device(config.device)

        if config.is_gguf:
            logger.info(f"[LTX] Cargando pipeline Distilled GGUF: {config.checkpoint_path}")
            pipeline = LTXDistilledGGUFVideoPipeline.create(
                checkpoint_path=config.checkpoint_path,
                gemma_root=config.gemma_root,
                upsampler_path=config.upsampler_path,
                device=torch_device,
                vae_video_path=config.vae_video_path,
                audio_vae_path=config.audio_vae_path,
                connector_path=config.connector_path,
            )
        else:
            logger.info(f"[LTX] Cargando pipeline Fast: {config.checkpoint_path}")
            pipeline = LTXFastVideoPipeline.create(
                checkpoint_path=config.checkpoint_path,
                gemma_root=config.gemma_root,
                upsampler_path=config.upsampler_path,
                device=torch_device,
                vae_video_path=config.vae_video_path,
                audio_vae_path=config.audio_vae_path,
                connector_path=config.connector_path,
            )

        return pipeline

    @staticmethod
    def run_inference(pipeline: Any, settings: LTXGenerationSettings) -> tuple:
        """
        Ejecuta la inferencia de video y retorna un tensor compatible con ComfyUI
        junto con audio extraído si está disponible.
        Returns:
            (video_tensor, audio_np, sampling_rate)
        """
        tiling_config = default_tiling_config()

        # Preparar condicionamiento de imagen (I2V) si aplica
        images_input: List[Dict[str, Any]] = []
        temp_file: Optional[str] = None

        if settings.use_image_conditioning:
            try:
                temp_file = LTXPipelineAdapter._prepare_image_conditioning(
                    image_path=settings.image_path,
                    width=settings.width,
                    height=settings.height,
                    strength=settings.strength,
                    seed=settings.seed,
                )
                images_input = [{"path": temp_file, "frame_idx": 0, "strength": settings.strength}]
            except Exception as e:
                logger.error(f"[LTX] Error preparando imagen condicional: {e}", exc_info=True)
                raise

        try:
            logger.info(
                f"[LTX] Inferencia: {settings.width}x{settings.height} @ {settings.frame_rate}fps "
                f"| frames={settings.num_frames} | seed={settings.seed} | I2V={settings.use_image_conditioning}"
            )

            result = pipeline._run_inference(
                prompt=settings.prompt,
                seed=settings.seed,
                height=settings.height,
                width=settings.width,
                num_frames=settings.num_frames,
                frame_rate=settings.frame_rate,
                images=images_input,
                tiling_config=tiling_config,
            )

            video_tensor, audio_np, sampling_rate = extract_video_audio(result)
            video_tensor = LTXPipelineAdapter._normalize_video(video_tensor)
            logger.info(f"[LTX] Inferencia completada. Video shape: {video_tensor.shape} | Audio: {audio_np is not None}")
            return video_tensor, audio_np, sampling_rate

        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                    logger.debug(f"[LTX] Archivo temporal eliminado: {temp_file}")
                except Exception as e:
                    logger.warning(f"[LTX] No se pudo eliminar temp file: {e}")

    @staticmethod
    def _prepare_image_conditioning(
        image_path: str, width: int, height: int, strength: float, seed: int
    ) -> str:
        """
        Carga una imagen desde disco, la redimensiona a las dimensiones de salida,
        y la guarda en un archivo temporal para el pipeline LTX.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"[LTX] Imagen condicional no encontrada: {image_path}")

        pil_image = Image.open(image_path).convert("RGB")
        pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)

        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, f"ltx_cond_hex_{seed}.png")
        pil_image.save(temp_file)
        logger.debug(f"[LTX] Imagen condicional preparada: {temp_file}")
        return temp_file

    @staticmethod
    def _normalize_video(video_tensor: torch.Tensor) -> torch.Tensor:
        """
        Normaliza un tensor de video a formato ComfyUI estándar:
        [frames, height, width, 3] float32 en rango [0.0, 1.0].
        """
        if video_tensor.dtype != torch.float32:
            video_tensor = video_tensor.to(dtype=torch.float32)

        if video_tensor.max() > 1.0:
            video_tensor = video_tensor / 255.0

        return video_tensor.clamp(0.0, 1.0)

    @staticmethod
    def _normalize_output(result: Any) -> torch.Tensor:
        """
        Convierte la salida del pipeline LTX a formato ComfyUI estándar.
        Deprecated: usar extract_video_audio + _normalize_video.
        """
        video_tensor, _audio, _sr = extract_video_audio(result)
        return LTXPipelineAdapter._normalize_video(video_tensor)
