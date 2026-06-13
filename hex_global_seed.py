# -*- coding: utf-8 -*-
import logging
try:
    from infrastructure.doc_adapter import hex_node_doc
except ImportError:
    from .infrastructure.doc_adapter import hex_node_doc

from .infrastructure.error_adapter import hex_error_handler

logger = logging.getLogger(__name__)

_CONTROL_VALUES = ["randomize", "fixed", "increment", "decrement"]


@hex_node_doc
class GlobalSeedHex:
    """
    [HEX] Global Seed.
    - control_after_generate (auto ComfyUI): controla el seed de este nodo.
    - mode: valor que se hereda a todos los FluxSamplerParametersHex al sincronizar.
    - auto_sync: cuando esta ON, sincroniza automaticamente al ejecutar.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "section_seed": ("STRING", {"default": "GLOBAL SEED"}),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "control_after_generate": True,
                }),
                "mode": (_CONTROL_VALUES, {"default": "fixed"}),
                "auto_sync": ("BOOLEAN", {"default": True, "label_on": "ON", "label_off": "OFF"}),
            }
        }

    RETURN_TYPES  = ("INT",)
    RETURN_NAMES  = ("seed",)
    OUTPUT_NODE   = True
    FUNCTION      = "execute"
    CATEGORY      = "flux_collection_advanced/hex"

    @hex_error_handler
    def execute(self, **kwargs):
        resolved = kwargs.get("seed", 0)
        mode     = kwargs.get("mode", "fixed")
        auto_sync = kwargs.get("auto_sync", True)

        logger.debug(f"[GlobalSeedHex] seed={resolved} mode={mode}")

        return {
            "ui": {
                "seed":      [resolved],
                "mode":      [mode],
                "auto_sync": [auto_sync],
            },
            "result": (resolved,)
        }
