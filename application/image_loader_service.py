# Task-Source: T#12-GGUF
# -*- coding: utf-8 -*-
"""
Servicio de aplicacion para la carga del stack completo Flux GGUF.
Orquesta el orden de carga sin conocer los detalles de implementacion.
"""
import logging
try:
    from domain.models import FluxModelConfig
    from infrastructure.image_model_adapter import FluxModelAdapter
    from infrastructure.error_adapter import ErrorLoggingAdapter
except ImportError:
    from ..domain.models import FluxModelConfig
    from ..infrastructure.image_model_adapter import FluxModelAdapter
    from ..infrastructure.error_adapter import ErrorLoggingAdapter

logger = logging.getLogger(__name__)


class FluxLoaderService:
    """
    Servicio de aplicacion que coordina la carga secuencial del stack Flux:
    UNET → sampling patch → CLIP → VAE.
    Recibe adaptadores por inyeccion de dependencias.
    """

    def __init__(self, model_adapter: FluxModelAdapter, error_adapter: ErrorLoggingAdapter):
        self.adapter = model_adapter
        self.error_adapter = error_adapter

    def load_full_flux_stack(self, config: FluxModelConfig) -> dict:
        """
        Carga el stack completo Flux en el orden correcto.

        Returns:
            dict con claves: "model", "clip", "vae"
        """
        logger.info(f"[Flux] Iniciando carga de stack: {config.unet_name} | base={config.base_type}")

        # 1. UNET
        model = self.adapter.load_unet(config)

        # 2. Patch de sampling (solo para modelos Flux nativos)
        if config.base_type in ("flux", "flux2"):
            self.adapter.apply_flux_sampling(model)

        # 3. CLIP
        clip = self.adapter.load_clip(config)

        # 4. VAE
        vae = self.adapter.load_vae(config.vae_name)

        logger.info(f"[Flux] Stack {config.base_type} cargado correctamente.")
        return {"model": model, "clip": clip, "vae": vae}
