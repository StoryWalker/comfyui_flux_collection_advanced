# -*- coding: utf-8 -*-
import torch
import logging
import node_helpers
import comfy.utils
from typing import Any, Dict, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WanSceneEncoder_Dev:
    """
    [DEV] Specialized Scene Encoder for Wan 2.2.
    Integrates Text Encoding and CLIP Vision Encoding in one step.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "clip_vision": ("CLIP_VISION",),
                "reference_image": ("IMAGE",),
                "positive_text": ("STRING", {"multiline": True, "default": "Cinematic video, high detail"}),
                "negative_text": ("STRING", {"multiline": True, "default": "blurry, worst quality"}),
                "width": ("INT", {"default": 480, "min": 16, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 848, "min": 16, "max": 2048, "step": 16}),
                "upscale_method": (["nearest-exact", "bilinear", "bicubic", "area", "lanczos"], {"default": "nearest-exact"}),
                "crop_position": (["center", "top", "bottom", "left", "right"], {"default": "center"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "IMAGE", "CLIP_VISION_OUTPUT",)
    RETURN_NAMES = ("positive", "negative", "image", "clip_vision_output",)
    FUNCTION = "encode_scene"
    CATEGORY = "flux_collection_advanced/_dev"

    def _prepare_vision(self, clip_vision, image):
        """ Replicates standard CLIP Vision encoding with internal crop """
        h, w = image.shape[1:3]
        length = min(h, w)
        y, x = (h - length) // 2, (w - length) // 2
        cropped = image[:, y:y+length, x:x+length, :]
        return clip_vision.encode_image(cropped, crop=False)

    def encode_scene(self, clip, clip_vision, reference_image, positive_text, negative_text, 
                     width, height, upscale_method, crop_position):
        
        logger.info("[DEV] Scene Encoder: Processing Vision and Text...")

        # 1. Resize Image
        img_in = reference_image.movedim(-1, 1)
        img_resized = comfy.utils.common_upscale(img_in, width, height, upscale_method, crop_position).movedim(1, -1)

        # 2. Vision Encode
        cv_out = self._prepare_vision(clip_vision, img_resized)

        # 3. Text Encode
        tokens_p = clip.tokenize(positive_text)
        cond_p, pooled_p = clip.encode_from_tokens(tokens_p, return_pooled=True)
        positive = [[cond_p, {"pooled_output": pooled_p}]]

        tokens_n = clip.tokenize(negative_text)
        cond_n, pooled_n = clip.encode_from_tokens(tokens_n, return_pooled=True)
        negative = [[cond_n, {"pooled_output": pooled_n}]]

        return (positive, negative, img_resized, cv_out)

# Registered via __init__.py
