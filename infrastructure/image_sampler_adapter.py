# -*- coding: utf-8 -*-
"""
Adaptador de infraestructura para el sampling Flux de etapa unica.
Encapsula comfy.sample, latent_preview y la decodificacion VAE con tiling.
"""
import logging
from typing import Any, Dict, Optional

import torch
import comfy.model_management
import comfy.samplers
import comfy.sample
import comfy.utils
import latent_preview

from ..domain.models import FluxSamplerConfig

logger = logging.getLogger(__name__)


class FluxSamplerAdapter:
    """
    Adaptador de infraestructura para sampling Flux.
    Separa la deteccion de arquitectura, la preparacion del latente,
    el sampling y la decodificacion VAE del codigo de orquestacion.
    """

    # ------------------------------------------------------------------
    # Generacion de latente vacio
    # ------------------------------------------------------------------

    @staticmethod
    def generate_empty_latent(config: FluxSamplerConfig, device: torch.device, channels: int, vae: Any = None) -> Dict[str, torch.Tensor]:
        compression = getattr(vae, "downscale_ratio", 8)
        latent = torch.zeros(
            [config.batch_size, channels, config.height // compression, config.width // compression],
            device=device,
        )
        logger.info(f"[Flux] Latente vacio generado: {latent.shape} en {device}")
        return {"samples": latent}

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @staticmethod
    def run_sample(model: Any, config: FluxSamplerConfig,
                   positive: Any, latent_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Ejecuta el sampling Flux de etapa unica con preparacion de ruido
        y callback de previsualizacion.
        """
        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_dict["samples"])
        model_device = model.load_device
        noise = comfy.sample.prepare_noise(latent_image, config.seed).to(model_device)
        preview_callback = latent_preview.prepare_callback(model, config.steps)

        samples = comfy.sample.sample(
            model=model,
            noise=noise,
            steps=config.steps,
            cfg=config.cfg,
            sampler_name=config.sampler_name,
            scheduler=config.scheduler,
            positive=positive,
            negative="",
            latent_image=latent_image.to(model_device),
            denoise=config.denoise,
            disable_noise=False,
            start_step=None,
            last_step=None,
            force_full_denoise=False,
            noise_mask=None,
            callback=preview_callback,
            disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
            seed=config.seed,
        )

        if samples is None:
            raise RuntimeError("[Flux] El sampler retorno None.")
        return samples

    # ------------------------------------------------------------------
    # Decodificacion VAE
    # ------------------------------------------------------------------

    @staticmethod
    def decode(vae: Any, samples: torch.Tensor, tiling_mode: str) -> torch.Tensor:
        """
        Decodifica el latente a imagen con soporte de tiling para alta resolucion.
        Maneja latentes 5D (video) aplanandolos antes del decode.
        """
        logger.info(f"[Flux] Decodificando VAE (tiling={tiling_mode})...")

        if hasattr(samples, "is_nested") and samples.is_nested:
            samples = samples.unbind()[0]

        original_shape = samples.shape
        is_video = len(original_shape) == 5
        if is_video:
            b, c, t, h, w = original_shape
            samples = samples.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)

        comfy.model_management.soft_empty_cache()

        pixels = vae.decode_tiled(samples) if tiling_mode == "enabled" else vae.decode(samples)

        if pixels is not None and len(pixels.shape) == 5:
            pixels = pixels.reshape(-1, pixels.shape[-3], pixels.shape[-2], pixels.shape[-1])

        logger.info(f"[Flux] Decode exitoso. Shape: {pixels.shape}")
        return pixels
