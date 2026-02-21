# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import folder_paths
import comfy.utils
from PIL import Image

class FluxImageComparison:
    """
    Flux Image Comparison (Standalone):
    Saves two images to temp and sends them to a custom JS handler.
    Does not inherit from PreviewImage to avoid default UI conflicts.
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
    FUNCTION = "compare"
    OUTPUT_NODE = True
    CATEGORY = "flux_collection_advanced"

    def compare(self, image_a, image_b):
        # Setup paths
        output_dir = folder_paths.get_temp_directory()
        subfolder = "flux_comparison"
        full_output_folder = os.path.join(output_dir, subfolder)
        os.makedirs(full_output_folder, exist_ok=True)

        results = []
        for i, img in enumerate([image_a, image_b]):
            # Convert first image in batch to PIL
            i_tensor = img[0]
            i_array = 255. * i_tensor.cpu().numpy()
            i_pil = Image.fromarray(np.clip(i_array, 0, 255).astype(np.uint8))
            
            filename = f"cmp_{'A' if i==0 else 'B'}_{os.urandom(4).hex()}.png"
            i_pil.save(os.path.join(full_output_folder, filename), compress_level=1)
            
            results.append({
                "filename": filename,
                "subfolder": subfolder,
                "type": "temp"
            })

        # Return custom key to avoid triggering ComfyUI's default preview
        return {"ui": {"flux_compare": results}}

# Registered via __init__.py
