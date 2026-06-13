# -*- coding: utf-8 -*-
import logging
from .infrastructure.persistence_adapter import FilePersistenceAdapter
from .infrastructure.error_adapter import hex_error_handler

logger = logging.getLogger(__name__)

class LoopStorageHex:
    """
    [HEX] v3.0.0 Loop Storage (The Sink).
    Receives the last frame from the sampler and saves it for the next run.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "section_storage": ("STRING", {"default": "LOOP STORAGE SETTINGS"}),
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "flux_collection_advanced/hex"
    DESCRIPTION = (
        "[HEX] v3.0.0 Loop Storage (The Sink).\nReceives the last frame from the sampler and saves it for the next run.\n\n"
        "This documentation was AI-generated. If you find any errors or have suggestions for "
        "improvement, please feel free to contribute! Edit on GitHub."
    )

    @hex_error_handler
    def execute(self, **kwargs):
        image = kwargs.get("image")
        if image is not None:
            persistence = FilePersistenceAdapter()
            persistence.save_frame(image)
            logger.info("[HEX] Loop Storage: Frame saved successfully.")
        return {}

# Registered via __init__.py
