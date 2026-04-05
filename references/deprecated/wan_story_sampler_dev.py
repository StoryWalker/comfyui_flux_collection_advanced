# -*- coding: utf-8 -*-
import torch
import logging
import os
import node_helpers
import comfy.utils
import comfy.model_management
import comfy.sample
import nodes
from typing import Any, Dict, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WanStorySampler_v2:
    """
    [DEV] v2.1.0 Isolated Atomic Sampler:
    Stand-alone implementation. No dependencies on other files.
    Fully aligned with Wan 2.2 Inpainting architecture.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "model_high": ("MODEL",),
                "model_low": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "clip_vision": ("CLIP_VISION",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "width": ("INT", {"default": 480, "min": 16, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 848, "min": 16, "max": 2048, "step": 16}),
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 241, "step": 4}),
                "steps_high": ("INT", {"default": 4, "min": 1, "max": 50}),
                "steps_low": ("INT", {"default": 4, "min": 1, "max": 50}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "upscale_method": (["nearest-exact", "bilinear", "bicubic", "area", "lanczos"], {"default": "nearest-exact"}),
                "crop_position": (["center", "top", "bottom", "left", "right"], {"default": "center"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("image", "last_image",)
    FUNCTION = "generate"
    CATEGORY = "flux_collection_advanced/_dev"

    def _center_crop(self, image):
        h, w = image.shape[1:3]
        length = min(h, w)
        y, x = (h - length) // 2, (w - length) // 2
        return image[:, y:y+length, x:x+length, :]

    def generate(self, **kwargs):
        # 1. Extraction & Safety
        ref_img = kwargs.get("reference_image")
        m_high = kwargs.get("model_high")
        m_low = kwargs.get("model_low") if kwargs.get("model_low") is not None else m_high
        vae, cv = kwargs.get("vae"), kwargs.get("clip_vision")
        pos, neg = kwargs.get("positive"), kwargs.get("negative")
        w, h, f = kwargs.get("width"), kwargs.get("height"), kwargs.get("num_frames")
        s_high, s_low, cfg, seed, dns = kwargs.get("steps_high"), kwargs.get("steps_low"), kwargs.get("cfg"), kwargs.get("seed"), kwargs.get("denoise")
        up_m, crop_p = kwargs.get("upscale_method"), kwargs.get("crop_position")

        device = comfy.model_management.get_torch_device()
        
        # 2. Image and Vision Preparation
        img_in = ref_img.movedim(-1, 1)
        img_resized = comfy.utils.common_upscale(img_in, w, h, up_m, crop_p).movedim(1, -1)
        # Deep vision anchor from resized image
        cv_out = cv.encode_image(self._center_crop(img_resized), crop=False)

        # 3. WAN 2.2 Concat Context Construction
        latent_t = ((f - 1) // 4) + 1
        latent = torch.zeros([1, 16, latent_t, h // 8, w // 8], device=device)
        
        # Build sequence sequence: Frame 0 is Ref, rest is 0.5
        sequence = torch.ones((f, h, w, 3), device=device, dtype=img_resized.dtype) * 0.5
        sequence[0] = img_resized[0]
        
        concat_img = vae.encode(sequence)
        concat_mask = torch.ones((1, 1, latent_t, h // 8, w // 8), device=device)
        concat_mask[:, :, :1] = 0.0 # Anchor first block
        
        c_vals = {"clip_vision_output": cv_out, "concat_latent_image": concat_img, "concat_mask": concat_mask}
        p_final = node_helpers.conditioning_set_values(pos, c_vals)
        n_final = node_helpers.conditioning_set_values(neg, c_vals)

        # 4. Sampling Loop
        total_steps = s_high + s_low
        l_dict = {"samples": latent}

        logger.info(f"[DEV] Atomic Story Sampler Phase 1: 0 to {s_high}")
        samples = nodes.common_ksampler(m_high, seed, total_steps, cfg, "lcm", "simple", p_final, n_final, l_dict, dns, False, 0, s_high, False)[0]
        
        logger.info(f"[DEV] Atomic Story Sampler Phase 2: {s_high} to {total_steps}")
        comfy.model_management.soft_empty_cache()
        samples = nodes.common_ksampler(m_low, seed, total_steps, cfg, "lcm", "simple", p_final, n_final, samples, dns, True, s_high, total_steps, True)[0]

        # 5. Decode
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        v_out = vae.decode(samples["samples"])
        if len(v_out.shape) == 5 and v_out.shape[0] == 1: v_out = v_out.squeeze(0)
        
        return (v_out, v_out[-1:].clone())

# Registration handled in __init__.py
