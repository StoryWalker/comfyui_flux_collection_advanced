# -*- coding: utf-8 -*-
import logging
from .infrastructure.image_io_adapter import ImageIOAdapter
from .infrastructure.error_adapter import hex_error_handler

logger = logging.getLogger(__name__)


class ImageComparisonHex:
    """
    [HEX] Image Comparison.
    Muestra dos imagenes lado a lado en el UI de ComfyUI para comparacion visual.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE  = True
    FUNCTION     = "execute"
    CATEGORY     = "flux_collection_advanced/hex"

    @hex_error_handler
    def execute(self, **kwargs):
        image_a = kwargs.get("image_a")
        image_b = kwargs.get("image_b")

        ref_a = ImageIOAdapter.save_temp_image(image_a, "a")
        ref_b = ImageIOAdapter.save_temp_image(image_b, "b")

        return {"ui": {"a": [ref_a], "b": [ref_b]}}
