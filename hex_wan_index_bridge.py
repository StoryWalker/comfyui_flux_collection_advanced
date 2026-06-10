# Task-Source: T#27
# -*- coding: utf-8 -*-
import logging
from .infrastructure.error_adapter import hex_error_handler

logger = logging.getLogger(__name__)


class WanIndexBridgeHex:
    """
    [HEX] Wan Index Bridge:
    Un puente simple para permitir que el Secuenciador devuelva su índice a sí mismo,
    evitando la restricción de bucle directo de ComfyUI.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "section_bridge": ("STRING", {"default": "WAN INDEX BRIDGE"}),
                "index_in": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("index_out",)
    FUNCTION = "execute"
    CATEGORY = "flux_collection_advanced/hex"

    @hex_error_handler
    def execute(self, **kwargs):
        index_in = kwargs["index_in"]
        return (index_in,)
