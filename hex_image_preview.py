# Task-Source: T#17
# -*- coding: utf-8 -*-
import os
import random
import json
import logging
import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths
from .infrastructure.error_adapter import hex_error_handler

logger = logging.getLogger(__name__)

try:
    from comfy import cli_args
except ImportError:
    class _MockArgs:
        disable_metadata = False
        disable_save_metadata = False
    cli_args = _MockArgs()

_RANDOM_CHARS = "abcdefghijklmnopqrstuvwxyz"


class FluxImagePreviewHex:
    """[HEX] Image Preview — genera previews temporales con metadatos opcionales."""

    RETURN_TYPES = ()
    FUNCTION     = "execute"
    OUTPUT_NODE  = True
    CATEGORY     = "flux_collection_advanced/hex"

    def __init__(self):
        self.output_dir     = folder_paths.get_temp_directory() or \
                              os.path.join(folder_paths.get_output_directory(), "temp")
        os.makedirs(self.output_dir, exist_ok=True)
        self.type           = "temp"
        self.prefix_append  = "_temp_" + "".join(random.choice(_RANDOM_CHARS) for _ in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "section_preview": ("STRING", {"default": "PREVIEW"}),
                "images": ("IMAGE", {"tooltip": "Imágenes a previsualizar en la UI."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    @hex_error_handler
    def execute(self, **kwargs):
        images       = kwargs["images"]
        prompt       = kwargs.get("prompt", None)
        extra_pnginfo = kwargs.get("extra_pnginfo", None)

        if images is None or images.shape[0] == 0:
            return {"ui": {"images": []}}

        prefix = "FluxPreview" + self.prefix_append
        h, w   = images.shape[1], images.shape[2]

        full_folder, filename, counter, subfolder, _ = \
            folder_paths.get_save_image_path(prefix, self.output_dir, h, w)

        metadata_disabled = getattr(cli_args, "disable_metadata", False) or \
                            getattr(cli_args, "disable_save_metadata", False)
        results = []

        for i, img_tensor in enumerate(images):
            arr = np.clip(255.0 * img_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
            pil = Image.fromarray(arr)

            meta = None
            if not metadata_disabled:
                meta = PngInfo()
                if prompt is not None:
                    meta.add_text("prompt", json.dumps(prompt))
                if isinstance(extra_pnginfo, dict):
                    for k, v in extra_pnginfo.items():
                        try:
                            meta.add_text(str(k), json.dumps(v))
                        except (TypeError, ValueError) as e:
                            logger.warning(f"[HEX] Preview: no se pudo serializar metadato '{k}': {e}")

            file = f"{filename.replace('%batch_num%', str(i))}_{counter:05}_.png"
            pil.save(os.path.join(full_folder, file), pnginfo=meta, compress_level=self.compress_level)
            results.append({"filename": file, "subfolder": subfolder, "type": self.type})
            counter += 1

        return {"ui": {"images": results}}

# Registrado via __init__.py
