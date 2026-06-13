# -*- coding: utf-8 -*-
import logging
import torch
from .domain.models import ImageLoadConfig
from .application.loader_service import ImageLoaderService
from .infrastructure.image_io_adapter import ImageIOAdapter
from .infrastructure.error_adapter import hex_error_handler

logger = logging.getLogger(__name__)

class ImageLoaderHex:
    """
    [HEX] v2.4.0 Universal Image Loader.
    Simplified widgets for stability.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        images = ImageIOAdapter.get_input_image_list()
        return {
            "required": {
                "section_load": ("STRING", {"default": "FILE SELECTION"}),
                "image": (images,),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "flux_collection_advanced/hex"
    DESCRIPTION = (
        "[HEX] v2.4.0 Universal Image Loader.\nSimplified widgets for stability.\n\n"
        "This documentation was AI-generated. If you find any errors or have suggestions for "
        "improvement, please feel free to contribute! Edit on GitHub."
    )

    @hex_error_handler
    def execute(self, **kwargs):
        image = kwargs.get("image", "")
        # 0. Validation: Check if it's a placeholder
        if not image or image.startswith("[") or "Error" in image:
            logger.warning("[HEX] Image Loader: No valid image selected.")
            blank = torch.zeros([1, 512, 512, 3])
            return {"ui": {"images": []}, "result": (blank,)}

        # 1. Create Domain Config
        config = ImageLoadConfig(image_path=image)

        # 2. Call Application Service
        service = ImageLoaderService()
        image_tensor = service.load_image(config)

        # 3. Return tensor and UI preview
        return {
            "ui": {
                "images": [
                    {
                        "filename": image,
                        "subfolder": "",
                        "type": "input"
                    }
                ]
            },
            "result": (image_tensor,)
        }

# Registered via __init__.py
