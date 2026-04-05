# -*- coding: utf-8 -*-
"""
Servicio de aplicacion para el sampling Flux de etapa unica.
Orquesta la preparacion del latente, el sampling y el decode.
"""
import logging
from typing import Any, Dict, Optional, Tuple

import torch
import comfy.model_management

from ..domain.models import FluxSamplerConfig
from ..infrastructure.image_sampler_adapter import FluxSamplerAdapter

logger = logging.getLogger(__name__)


class FluxSamplerService:
    """
    Coordina el pipeline de generacion Flux:
      1. Preparacion del latente (vacio o Img2Img)
      2. Sampling  (comfy.sample.sample maneja el conditioning internamente)
      3. Decodificacion VAE con soporte de tiling
    """

    def __init__(self, adapter: FluxSamplerAdapter):
        self.adapter = adapter

    def generate(
        self,
        config: FluxSamplerConfig,
        model: Any,
        positive: Any,
        vae: Any,
        latent_opt: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Ejecuta el pipeline completo.

        Returns:
            (image tensor [B, H, W, C], latent dict {"samples": tensor})
        """
        logger.info(f"[Flux] Iniciando pipeline: {config.width}x{config.height}, "
                    f"steps={config.steps}, seed={config.seed}")

        # 1. Latente
        if latent_opt is not None:
            logger.info("[Flux] Modo Img2Img: usando latente de entrada.")
            initial_latent = latent_opt
        else:
            device = comfy.model_management.get_torch_device()
            vae_channels = getattr(vae, "latent_channels", 16)
            initial_latent = self.adapter.generate_empty_latent(config, device, vae_channels, vae)

        # 2. Sampling
        sampled = self.adapter.run_sample(model, config, positive, initial_latent)
        out_latent = {"samples": sampled}

        # 3. Decode
        image = self.adapter.decode(vae, sampled, config.vae_tiling)

        return image, out_latent
