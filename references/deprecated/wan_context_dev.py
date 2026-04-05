# -*- coding: utf-8 -*-
import torch
import logging
import node_helpers
import comfy.utils
from typing import Any, Dict, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WanContext_Dev:
    """
    [DEV] Specialized Context Node for Wan 2.2.
    Isolated logic for building temporal concatenation latent and masks.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "reference_image": ("IMAGE",),
                "width": ("INT", {"default": 832, "min": 16, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 480, "min": 16, "max": 2048, "step": 16}),
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 241, "step": 4}),
                "upscale_method": (["nearest-exact", "bilinear", "bicubic", "area", "lanczos"], {"default": "nearest-exact"}),
                "crop_position": (["center", "top", "bottom", "left", "right"], {"default": "center"}),
            },
            "optional": {
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT",)
    RETURN_NAMES = ("positive", "negative", "latent",)
    FUNCTION = "prepare_context"
    CATEGORY = "flux_collection_advanced/_dev"

    def prepare_context(self, positive, negative, vae, reference_image, width, height, num_frames, 
                        upscale_method, crop_position, clip_vision_output=None):
        
        device = comfy.model_management.get_torch_device()
        
        # 1. Resize
        img_in = reference_image.movedim(-1, 1)
        img_resized = comfy.utils.common_upscale(img_in, width, height, upscale_method, crop_position).movedim(1, -1)

        # 2. Build 5D Latent Context
        latent_t = ((num_frames - 1) // 4) + 1
        
        # Base Latent: All Zeros
        latent = torch.zeros([1, 16, latent_t, height // 8, width // 8], device=device)
        
        # Concat Sequence: Frame 0 is Ref, others are grey
        sequence = torch.ones((num_frames, height, width, 3), device=device, dtype=img_resized.dtype) * 0.5
        sequence[0] = img_resized[0]
        
        # Encode
        concat_latent_image = vae.encode(sequence[:, :, :, :3])
        
        # Concat Mask: 0.0 for block 0, 1.0 rest
        concat_mask = torch.ones((1, 1, latent_t, height // 8, width // 8), device=device)
        concat_mask[:, :, :1] = 0.0 
        
        # Apply to Conditioning
        cond_values = {
            "concat_latent_image": concat_latent_image,
            "concat_mask": concat_mask
        }
        
        if clip_vision_output is not None:
            cond_values["clip_vision_output"] = clip_vision_output

        p_final = node_helpers.conditioning_set_values(positive, cond_values)
        n_final = node_helpers.conditioning_set_values(negative, cond_values)

        return (p_final, n_final, {"samples": latent})

# Registration info handled in __init__.py
