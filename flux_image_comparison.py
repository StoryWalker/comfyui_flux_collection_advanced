# -*- coding: utf-8 -*-
import os
import random
import numpy as np
import torch
import logging
from PIL import Image
import folder_paths

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FluxImageComparison:
    """
    Flux Image Comparison Node:
    Receives two images (A and B) and provides a side-by-side comparison slider in the UI.
    Ideal for comparing Upscale vs Original or Detailer vs Base.
    """
    
    RETURN_TYPES = ()
    FUNCTION = "compare_images"
    OUTPUT_NODE = True
    CATEGORY = "flux_collection_advanced"

    _DEFAULT_PREFIX = "FluxCompare"
    _RANDOM_CHARS = "abcdefghijklmnopqrstuvwxyz"
    _RANDOM_LENGTH = 5

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        if not self.output_dir:
            self.output_dir = os.path.join(folder_paths.get_output_directory(), "temp")
        
        os.makedirs(self.output_dir, exist_ok=True)
        self.type = "temp"
        self.prefix_append = "_cmp_" + ''.join(random.choice(self._RANDOM_CHARS) for _ in range(self._RANDOM_LENGTH))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE", {"tooltip": "Left image (Original/Reference)."}),
                "image_b": ("IMAGE", {"tooltip": "Right image (Upscaled/Detailed)."}),
            }
        }

    def compare_images(self, image_a: torch.Tensor, image_b: torch.Tensor):
        """
        Saves both images and returns metadata for the custom JS comparison widget.
        """
        results = []
        
        # We only take the first image of each batch for comparison to keep it simple
        for i, img_tensor in enumerate([image_a, image_b]):
            try:
                # Basic batch handling: take index 0
                tensor = img_tensor[0] if img_tensor.ndim == 4 else img_tensor
                
                # Convert to PIL
                img_array = 255. * tensor.cpu().numpy()
                img_array_clipped = np.clip(img_array, 0, 255).astype(np.uint8)
                img_pil = Image.fromarray(img_array_clipped)

                # Generate path
                filename_prefix = f"{self._DEFAULT_PREFIX}_{'A' if i==0 else 'B'}{self.prefix_append}"
                full_output_folder, filename, counter, subfolder, _ = \
                    folder_paths.get_save_image_path(filename_prefix, self.output_dir, img_pil.height, img_pil.width)

                file = f"{filename}_{counter:05}_.png"
                full_path = os.path.join(full_output_folder, file)
                
                # Save
                img_pil.save(full_path, compress_level=self.compress_level)

                results.append({
                    "filename": file,
                    "subfolder": subfolder,
                    "type": self.type,
                    "label": "Image A (Left)" if i == 0 else "Image B (Right)"
                })

            except Exception as e:
                logger.error(f"Failed to process comparison image {'A' if i==0 else 'B'}: {e}")
                raise RuntimeError(f"Comparison failed: {e}")

        # Return both images in the UI metadata
        return {"ui": {"images": results}}

# Registered via __init__.py
