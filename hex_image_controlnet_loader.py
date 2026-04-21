# Task-Source: T#15
# -*- coding: utf-8 -*-
import logging
import comfy.controlnet
import folder_paths
from .infrastructure.error_adapter import hex_error_handler

logger = logging.getLogger(__name__)


class FluxControlNetLoaderHex:
    """[HEX] ControlNet Loader — carga un modelo ControlNet compatible con Flux."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "section_config": ("STRING", {"default": "CONTROLNET LOADER"}),
                "model": ("MODEL", {"tooltip": "Diffusion model (UNET) al que se aplicará el ControlNet."}),
                "control_net_name": (folder_paths.get_filename_list("controlnet"), {}),
            }
        }

    RETURN_TYPES = ("CONTROL_NET",)
    RETURN_NAMES = ("control_net",)
    FUNCTION = "execute"
    CATEGORY = "flux_collection_advanced/hex"

    @hex_error_handler
    def execute(self, **kwargs):
        model = kwargs["model"]
        name  = kwargs["control_net_name"]
        path  = folder_paths.get_full_path_or_raise("controlnet", name)
        logger.info(f"[HEX] ControlNet Loader: cargando {name}")
        controlnet = comfy.controlnet.load_controlnet(path, model)
        return (controlnet,)

# Registrado via __init__.py
