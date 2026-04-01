# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from PIL import Image
import folder_paths

class FluxImageComparison:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image_a": ("IMAGE",), "image_b": ("IMAGE",)}}
    
    RETURN_TYPES = ()
    FUNCTION = "compare"
    OUTPUT_NODE = True
    CATEGORY = "flux_collection_advanced"

    def compare(self, image_a, image_b):
        out = {"ui": {"a":[], "b":[]}}
        for key, img in [("a", image_a), ("b", image_b)]:
            # Extraer primera imagen del batch
            i = 255. * img[0].cpu().numpy()
            img_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            
            temp_path = folder_paths.get_temp_directory()
            filename = f"cmp_{key}_{os.urandom(2).hex()}.png"
            img_pil.save(os.path.join(temp_path, filename), compress_level=1)
            
            out["ui"][key].append({"filename": filename, "type": "temp", "subfolder": ""})
        
        return out
