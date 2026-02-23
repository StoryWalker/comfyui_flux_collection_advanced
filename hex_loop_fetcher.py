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
    [HEX] v3.2.0 Robust Loop Fetcher.
    Forces cache bypass to ensure sequential updates during batch runs.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "section_loop": ("STRING", {"default": "LOOP CONFIGURATION"}),
                "mode": (["Initial Frame", "Continue from Disk"], {"default": "Initial Frame"}),
            },
            "optional": {
                "initial_image": ("IMAGE",),
                # Trick: This hidden input forces ComfyUI to re-execute the node 
                # if we could connect it, but for now we'll use time-based logic.
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE_OUT",)
    FUNCTION = "execute"
    CATEGORY = "flux_collection_advanced/hex"

    # Crucial: IS_CHANGED forces ComfyUI to re-run this node every time Queue is pressed
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()

    @hex_error_handler
    def execute(self, **kwargs):
        mode = kwargs.get("mode", "Initial Frame")
        initial_image = kwargs.get("initial_image", None)
        
        logger.info(f"[HEX] Loop Fetcher: Fetching frame for mode '{mode}'...")
        
        config = BufferConfig(mode=mode)
        service = ContinuityService()
        
        # Sincronización: Forzar lectura fresca del disco
        image_out = service.sync_image(config, initial_image=initial_image, sampler_last_image=None)

        return (image_out,)

# Registered via __init__.py
