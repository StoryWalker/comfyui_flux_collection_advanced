# Task-Source: T#4
# -*- coding: utf-8 -*-
import logging
import torch
import time
from .domain.models import BufferConfig
from .application.continuity_service import ContinuityService
from .infrastructure.error_adapter import hex_error_handler

logger = logging.getLogger(__name__)

class LoopFetcherHex:
    """
    [HEX] v3.2.1 Robust Loop Fetcher.
    Forces cache bypass to ensure sequential updates during batch runs.
    Las dimensiones de fallback se configuran aqui para eliminar valores hardcodeados
    en la capa de aplicacion.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "section_loop": ("STRING", {"default": "LOOP CONFIGURATION"}),
                "mode": (["Initial Frame", "Continue from Disk"], {"default": "Initial Frame"}),
                "fallback_width":  ("INT", {"default": 848, "min": 64, "max": 4096, "step": 8}),
                "fallback_height": ("INT", {"default": 480, "min": 64, "max": 4096, "step": 8}),
            },
            "optional": {
                "initial_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE_OUT",)
    FUNCTION = "execute"
    CATEGORY = "flux_collection_advanced/hex"
    DESCRIPTION = (
        "[HEX] v3.2.1 Robust Loop Fetcher.\nForces cache bypass to ensure sequential updates during batch runs.\nLas dimensiones de fallback se configuran aqui para eliminar valores hardcodeados\nen la capa de aplicacion.\n\n"
        "This documentation was AI-generated. If you find any errors or have suggestions for "
        "improvement, please feel free to contribute! Edit on GitHub."
    )

    # IS_CHANGED fuerza a ComfyUI a re-ejecutar este nodo cada vez que se presiona Queue
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()

    @hex_error_handler
    def execute(self, **kwargs):
        mode = kwargs.get("mode", "Initial Frame")
        initial_image = kwargs.get("initial_image", None)
        fallback_width = kwargs.get("fallback_width", 848)
        fallback_height = kwargs.get("fallback_height", 480)

        logger.info(f"[HEX] Loop Fetcher: modo '{mode}', fallback {fallback_width}x{fallback_height}")

        config = BufferConfig(mode=mode)
        service = ContinuityService()

        # Sincronizacion: Forzar lectura fresca del disco
        image_out = service.sync_image(
            config,
            initial_image=initial_image,
            sampler_last_image=None,
            fallback_height=fallback_height,
            fallback_width=fallback_width,
        )

        return (image_out,)

# Registered via __init__.py
