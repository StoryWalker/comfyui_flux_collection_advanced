# -*- coding: utf-8 -*-
"""
[NVFP4] Image Sampler — Nodo experimental de sampling unificado.
Combina text encoding + sampling + VAE decode en un solo nodo para validación rápida.
"""
import logging
import comfy.samplers

from comfy.comfy_types import IO

try:
    from domain.models import FluxSamplerConfig, TextEncodingConfig
    from application.nvfp4_image_generation_service import NVFP4ImageGenerationService
    from infrastructure.nvfp4_model_adapter import NVFP4ModelAdapter
    from infrastructure.clip_encoding_adapter import ClipEncodingAdapter
    from infrastructure.image_sampler_adapter import FluxSamplerAdapter
    from infrastructure.error_adapter import hex_error_handler
except ImportError:
    from .domain.models import FluxSamplerConfig, TextEncodingConfig
    from .application.nvfp4_image_generation_service import NVFP4ImageGenerationService
    from .infrastructure.nvfp4_model_adapter import NVFP4ModelAdapter
    from .infrastructure.clip_encoding_adapter import ClipEncodingAdapter
    from .infrastructure.image_sampler_adapter import FluxSamplerAdapter
    from .infrastructure.error_adapter import hex_error_handler

logger = logging.getLogger(__name__)


class NVFP4ImageSampler:
    """
    [NVFP4] Image Sampler:
    Toma model/clip/vae + prompt y genera la imagen final.
    Integra encoding + sampling + decode para flujo de validación rápida.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "section_prompt": ("STRING", {"default": "PROMPT"}),
                "prompt": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "default": "a futuristic city at sunset, high detail, 8k"}),

                "section_model": ("STRING", {"default": "MODEL INPUTS"}),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),

                "section_sampling": ("STRING", {"default": "SAMPLING"}),
                "width": ("INT", {"default": 1024, "min": 16, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 4096, "step": 16}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": (IO.FLOAT, {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
            },
            "optional": {
                "guidance": (IO.FLOAT, {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "execute"
    CATEGORY = "flux_collection_advanced/nvfp4"

    @hex_error_handler
    def execute(self, **kwargs):
        text = kwargs.get("prompt", "")
        guidance = kwargs.get("guidance", 3.5)

        text_config = TextEncodingConfig(
            text=text,
            style1="No Style",
            style2="No Style",
            style3="No Style",
            style4="No Style",
            guidance=guidance,
        )

        sampler_config = FluxSamplerConfig(
            width=kwargs.get("width", 1024),
            height=kwargs.get("height", 1024),
            batch_size=1,
            seed=kwargs.get("seed", 0),
            steps=kwargs.get("steps", 20),
            cfg=kwargs.get("cfg", 1.0),
            sampler_name=kwargs.get("sampler_name"),
            scheduler=kwargs.get("scheduler"),
            denoise=1.0,
            vae_tiling="enabled",
        )

        service = NVFP4ImageGenerationService(
            model_adapter=NVFP4ModelAdapter(),
            clip_adapter=ClipEncodingAdapter(),
            sampler_adapter=FluxSamplerAdapter(),
        )

        # 1. Encode prompt
        positive = service.encode_prompt(text_config, kwargs.get("clip"))

        # 2. Sample
        image, latent = service.sample(
            sampler_config,
            kwargs.get("model"),
            positive,
            kwargs.get("vae"),
        )

        logger.info(f"[NVFP4] Generación completada: {sampler_config.width}x{sampler_config.height}")
        return (image, latent)
