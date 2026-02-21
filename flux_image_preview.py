# -*- coding: utf-8 -*-
import os
import random
import json
import numpy as np
import torch
import logging
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Handle CLI args for metadata disabling
try:
    from comfy import cli_args
except ImportError:
    class MockArgs:
        disable_metadata = False
        disable_save_metadata = False
    cli_args = MockArgs()

class FluxImagePreview:
    """
    Optimized Preview Node for Flux Collection Advanced:
    Generates high-speed temporary preview images with optional metadata embedding.
    Standardized logging and removed verbose debug prints for cleaner console output.
    """
    
    RETURN_TYPES = ()
    FUNCTION = "preview_images"
    OUTPUT_NODE = True
    CATEGORY = "flux_collection_advanced"

    _DEFAULT_PREFIX = "FluxPreview"
    _RANDOM_CHARS = "abcdefghijklmnopqrstuvwxyz"
    _RANDOM_LENGTH = 5

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        if not self.output_dir:
            self.output_dir = os.path.join(folder_paths.get_output_directory(), "temp")
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice(self._RANDOM_CHARS) for _ in range(self._RANDOM_LENGTH))
        self.compress_level = 1 # High speed for preview

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to preview in the UI."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    def preview_images(self, images: torch.Tensor, prompt=None, extra_pnginfo=None):
        """
        Saves images to temp directory and returns metadata for UI display.
        Optimized for performance and clean console logging.
        """
        if images is None or images.shape[0] == 0:
            return {"ui": {"images": []}}

        filename_prefix = self._DEFAULT_PREFIX + self.prefix_append
        image_height, image_width = images.shape[1:3]
        
        # Resolve saving paths via ComfyUI standards
        try:
            full_output_folder, filename, counter, subfolder, filename_prefix_resolved = \
                folder_paths.get_save_image_path(filename_prefix, self.output_dir, image_height, image_width)
        except Exception as e:
            logger.error(f"Failed to resolve preview path: {e}")
            raise RuntimeError(f"Preview path resolution failed: {e}")

        results = []
        
        # Metadata check (CLI flags)
        metadata_disabled = getattr(cli_args, 'disable_metadata', False) or \
                            getattr(cli_args, 'disable_save_metadata', False)

        for i, image_tensor in enumerate(images):
            try:
                # Convert tensor to PIL
                img_array = 255. * image_tensor.cpu().numpy()
                img_array_clipped = np.clip(img_array, 0, 255).astype(np.uint8)
                img_pil = Image.fromarray(img_array_clipped)

                # Prepare Metadata (only if enabled)
                metadata = None
                if not metadata_disabled:
                    metadata = PngInfo()
                    if prompt is not None:
                        metadata.add_text("prompt", json.dumps(prompt))
                    if isinstance(extra_pnginfo, dict):
                        for key, value in extra_pnginfo.items():
                            try:
                                json_value = json.dumps(value)
                                if len(json_value) < 10000: # Limit metadata size
                                    metadata.add_text(str(key), json_value)
                            except: pass

                # Generate unique temp filename
                file = f"{filename.replace('%batch_num%', str(i))}_{counter:05}_.png"
                full_path = os.path.join(full_output_folder, file)
                
                # Save as temporary PNG
                img_pil.save(full_path, pnginfo=metadata, compress_level=self.compress_level)

                results.append({
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self.type
                })
                counter += 1

            except Exception as e:
                logger.warning(f"Failed to process preview image at index {i}: {e}")
                continue

        return {"ui": {"images": results}}

# Registered via __init__.py
