# -*- coding: utf-8 -*-
import logging
from .domain.models import VideoExportManifest
from .application.export_service import ExportVideoService
from .infrastructure.error_adapter import hex_error_handler

logger = logging.getLogger(__name__)

class WanVideoSaverHex:
    """
    [HEX] v2.6.0 Saver strictly aligned with JSON indices.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "section_export": ("STRING", {"default": "EXPORT CONFIGURATION"}),
                "images": ("IMAGE",), # Socket 0
                "filename_prefix": ("STRING", {"default": "Scene"}), # Socket 1
                "custom_path": ("STRING", {"default": ""}), # Socket 2
                "fps": ("INT", {"default": 25, "min": 1, "max": 120}), # Socket 3
                "index": ("INT", {"default": 0, "min": 0, "max": 9999}), # Socket 4
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "flux_collection_advanced/hex"

    @hex_error_handler
    def execute(self, **kwargs):
        images = kwargs.get("images")
        if images is None:
            return {}

        manifest = VideoExportManifest(
            filename_prefix=kwargs.get("filename_prefix", "Scene"),
            index=kwargs.get("index", 0),
            fps=kwargs.get("fps", 25),
            custom_path=kwargs.get("custom_path", "")
        )

        service = ExportVideoService()
        service.export_video(manifest, images)
        return {}
