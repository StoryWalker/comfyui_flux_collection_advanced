# -*- coding: utf-8 -*-
import logging
from nodes import PreviewImage

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FluxImageComparison(PreviewImage):
    """
    Flux Image Comparison:
    Basado en la estructura de Rgthree Image Comparer para máxima compatibilidad.
    """
    FUNCTION = "compare_images"
    CATEGORY = "flux_collection_advanced"
    OUTPUT_NODE = True
    RETURN_TYPES = ()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image_a": ("IMAGE", {"tooltip": "Imagen Izquierda (A)"}),
                "image_b": ("IMAGE", {"tooltip": "Imagen Derecha (B)"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    def compare_images(self, image_a=None, image_b=None, prompt=None, extra_pnginfo=None):
        prefix_a = "flux.compare.a."
        prefix_b = "flux.compare.b."

        result = {"ui": {"a_images": [], "b_images": []}}
        
        if image_a is not None and len(image_a) > 0:
            result['ui']['a_images'] = self.save_images(image_a, prefix_a, prompt, extra_pnginfo)['ui']['images']

        if image_b is not None and len(image_b) > 0:
            result['ui']['b_images'] = self.save_images(image_b, prefix_b, prompt, extra_pnginfo)['ui']['images']

        return result

# Registro en __init__.py ya existente
