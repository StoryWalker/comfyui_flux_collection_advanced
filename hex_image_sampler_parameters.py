# -*- coding: utf-8 -*-
import logging
try:
    from infrastructure.doc_adapter import hex_node_doc
except ImportError:
    from .infrastructure.doc_adapter import hex_node_doc


import comfy.samplers
import nodes

from .domain.models import FluxSamplerConfig
from .application.image_sampler_service import FluxSamplerService
from .infrastructure.image_sampler_adapter import FluxSamplerAdapter
from .infrastructure.error_adapter import hex_error_handler

logger = logging.getLogger(__name__)


@hex_node_doc
class FluxSamplerParametersHex:
    """
    [HEX] Flux Sampler Parameters.
    Sampler Flux avanzado con VAE Tiling para alta resolucion
    y soporte de latente opcional para Img2Img/Refinement.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":    ("MODEL",),
                "positive": ("CONDITIONING",),
                "vae":      ("VAE",),

                # --- SECTION: IMAGE SETTINGS ---
                "section_image": ("STRING", {"default": "IMAGE SETTINGS"}),
                "width":      ("INT", {"default": 512, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height":     ("INT", {"default": 512, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),

                # --- SECTION: SAMPLING ---
                "section_sampling": ("STRING", {"default": "SAMPLING PARAMETERS"}),
                "seed":         ("INT",   {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps":        ("INT",   {"default": 20, "min": 1, "max": 10000}),
                "cfg":          ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01,
                                           "tooltip": "CFG scale (1.0 es el estandar para Flux)."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler":    (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise":      ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, 
                                           "tooltip": "Mantener en 1.0 para Txt2Img y Kontext. Bajar (ej. 0.7) solo al usar latent_opt (Img2Img clásico)."}),

                # --- SECTION: OPTIONS ---
                "section_options": ("STRING", {"default": "OPTIONS"}),
                "vae_tiling": (["enabled", "disabled"], {"default": "enabled",
                                "tooltip": "Tiled VAE decoding para 2K/4K sin OOM."}),
            },
            "optional": {
                "latent_opt": ("LATENT", {"tooltip": "Latente opcional para modo Img2Img tradicional."}),
                "reference_opt": ("LATENT", {"tooltip": "Latente de referencia para Kontext/Redux (se inyecta como condicionamiento)."}),
            },
        }

    RETURN_TYPES  = ("IMAGE", "LATENT")
    RETURN_NAMES  = ("image", "latent")
    FUNCTION      = "execute"
    CATEGORY      = "flux_collection_advanced/hex"

    @hex_error_handler
    def execute(self, **kwargs):
        config = FluxSamplerConfig(
            width        = kwargs.get("width", 1024),
            height       = kwargs.get("height", 1024),
            batch_size   = kwargs.get("batch_size", 1),
            seed         = kwargs.get("seed", 0),
            steps        = kwargs.get("steps", 28),
            cfg          = kwargs.get("cfg", 1.0),
            sampler_name = kwargs.get("sampler_name"),
            scheduler    = kwargs.get("scheduler"),
            denoise      = kwargs.get("denoise", 1.0),
            vae_tiling   = kwargs.get("vae_tiling", "enabled"),
        )

        service = FluxSamplerService(adapter=FluxSamplerAdapter())
        image, latent = service.generate(
            config,
            model         = kwargs.get("model"),
            positive      = kwargs.get("positive"),
            vae           = kwargs.get("vae"),
            latent_opt    = kwargs.get("latent_opt"),
            reference_opt = kwargs.get("reference_opt"),
        )
        return (image, latent)
