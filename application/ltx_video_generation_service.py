# -*- coding: utf-8 -*-
"""
Application Service:
Orchestrates LTX Video 2.3 generation using Domain models and Infrastructure adapters.
"""
import logging
from typing import Any

try:
    from domain.models import LTXGenerationSettings
    from infrastructure.ltx_pipeline_adapter import LTXPipelineAdapter
    from infrastructure.error_adapter import ErrorLoggingAdapter
except ImportError:
    from ..domain.models import LTXGenerationSettings
    from ..infrastructure.ltx_pipeline_adapter import LTXPipelineAdapter
    from ..infrastructure.error_adapter import ErrorLoggingAdapter

logger = logging.getLogger(__name__)


class LTXVideoGenerationService:
    """
    Servicio de aplicación que orquesta la generación de video con LTX 2.3.
    Recibe adaptadores por inyección de dependencias para mantener testabilidad.
    """

    def __init__(
        self,
        pipeline_adapter: LTXPipelineAdapter = None,
        error_adapter: ErrorLoggingAdapter = None,
    ):
        self.pipeline_adapter = pipeline_adapter if pipeline_adapter is not None else LTXPipelineAdapter()
        self.error_adapter = error_adapter if error_adapter is not None else ErrorLoggingAdapter()

    def generate(self, pipeline: Any, settings: LTXGenerationSettings) -> tuple:
        """
        Ejecuta la inferencia de video LTX a partir de un pipeline cargado y settings de dominio.

        Args:
            pipeline: Instancia de pipeline LTX ya cargada (LTXFastVideoPipeline o LTXDistilledGGUFVideoPipeline).
            settings: Configuración de generación validada del dominio.

        Returns:
            (video_tensor, audio_data)
            video_tensor: [frames, height, width, 3] float32 [0.0, 1.0]
            audio_data: tuple(audio_np, sampling_rate) o None
        """
        logger.info(
            f"[HEX-LTX] Iniciando generación: {settings.width}x{settings.height} "
            f"| frames={settings.num_frames} | fps={settings.frame_rate} | seed={settings.seed}"
        )

        try:
            video_output, audio_np, sampling_rate = self.pipeline_adapter.run_inference(pipeline, settings)
            logger.info(f"[HEX-LTX] Generación completada. Video shape: {video_output.shape} | Audio: {audio_np is not None}")
            audio_data = (audio_np, sampling_rate) if audio_np is not None else None
            return video_output, audio_data
        except Exception as e:
            logger.error(f"[HEX-LTX] Fallo en generación LTX: {e}", exc_info=True)
            self.error_adapter.log_error(
                node_name="LTXVideoGenerationService",
                exception=e,
                context={
                    "width": settings.width,
                    "height": settings.height,
                    "num_frames": settings.num_frames,
                    "frame_rate": settings.frame_rate,
                    "seed": settings.seed,
                    "pipeline_type": getattr(pipeline, "pipeline_kind", "unknown"),
                },
            )
            raise
