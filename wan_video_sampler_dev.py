# -*- coding: utf-8 -*-
import torch
import logging
import os
import nodes
import node_helpers
import comfy.utils
import comfy.model_management
import comfy.sample
from typing import Any, Dict, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WanVideoSampler_Dev:
    """
    [DEV] v1.19 Extreme Stability Sampler for Wan 2.2.
    Fixed: 'Black screen' in long chains by implementing aggressive VRAM management.
    Features: Component passthrough and frame extraction for storytelling.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE", {"tooltip": "Starting frame."}),
                "model_high": ("MODEL",),
                "model_low": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "clip_vision": ("CLIP_VISION",),
                "positive_prompt": ("STRING", {"multiline": True, "default": "Cinematic video"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "worst quality"}),
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

    RETURN_TYPES = ("IMAGE", "IMAGE", "ANY", "CLIP", "VAE", "CLIP_VISION",)
    RETURN_NAMES = ("image", "last_image", "any", "clip", "vae", "clip_vision",)
    FUNCTION = "generate_video"
    CATEGORY = "flux_collection_advanced/_dev"

    def _prepare_vision(self, clip_vision, image):
        h, w = image.shape[1:3]
        length = min(h, w)
        y, x = (h - length) // 2, (w - length) // 2
        cropped = image[:, y:y+length, x:x+length, :]
        return clip_vision.encode_image(cropped, crop=False)

    def _encode_prompt(self, clip, prompt):
        tokens = clip.tokenize(prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]

    def generate_video(self, **kwargs):
        # 1. Extraction
        ref_img = kwargs.get("reference_image")
        m_high, m_low, clip, vae, cv = kwargs.get("model_high"), kwargs.get("model_low"), kwargs.get("clip"), kwargs.get("vae"), kwargs.get("clip_vision")
        p_str, n_str = kwargs.get("positive_prompt"), kwargs.get("negative_prompt")
        w, h, f = kwargs.get("width"), kwargs.get("height"), kwargs.get("num_frames")
        s_high, s_low, cfg, seed, dns = kwargs.get("steps_high"), kwargs.get("steps_low"), kwargs.get("cfg"), kwargs.get("seed"), kwargs.get("denoise")
        up_m, crop_p = kwargs.get("upscale_method"), kwargs.get("crop_position")

        device = comfy.model_management.get_torch_device()
        
        # 2. Preparation with cache clearing
        comfy.model_management.soft_empty_cache()
        img_in = ref_img.movedim(-1, 1)
        img_resized = comfy.utils.common_upscale(img_in, w, h, up_m, crop_p).movedim(1, -1)
        
        cv_out = self._prepare_vision(cv, img_resized)
        pos = self._encode_prompt(clip, p_str)
        neg = self._encode_prompt(clip, n_str)

        # 3. Context
        latent_t = ((f - 1) // 4) + 1
        latent = torch.zeros([1, 16, latent_t, h // 8, w // 8], device=device)
        sequence = torch.ones((f, h, w, 3), device=device, dtype=img_resized.dtype) * 0.5
        sequence[0] = img_resized[0]
        concat_img = vae.encode(sequence)
        
        concat_mask = torch.ones((1, 1, latent_t, h // 8, w // 8), device=device)
        concat_mask[:, :, :1] = 0.0 
        
        c_vals = {"clip_vision_output": cv_out, "concat_latent_image": concat_img, "concat_mask": concat_mask}
        p_final = node_helpers.conditioning_set_values(pos, c_vals)
        n_final = node_helpers.conditioning_set_values(neg, c_vals)

        # 4. Sampling Stage 1
        total_steps = s_high + s_low
        l_dict = {"samples": latent}

        logger.info(f"[DEV] Phase 1 (Structural): 0 to {s_high}")
        samples = nodes.common_ksampler(
            m_high, seed, total_steps, cfg, "lcm", "simple", 
            p_final, n_final, l_dict, denoise=dns, 
            disable_noise=False, start_step=0, last_step=s_high, force_full_denoise=False
        )[0]

        # VITAL: Unload model A before loading model B
        comfy.model_management.soft_empty_cache()

        # 5. Sampling Stage 2
        logger.info(f"[DEV] Phase 2 (Textural): {s_high} to {total_steps}")
        samples = nodes.common_ksampler(
            m_low, seed, total_steps, cfg, "lcm", "simple", 
            p_final, n_final, samples, denoise=dns, 
            disable_noise=True, start_step=s_high, last_step=total_steps, force_full_denoise=True
        )[0]

        # 6. EXTREME VRAM CLEANUP before VAE Decode
        # This prevents the black screens in step 3+
        logger.info("[DEV] Extreme VRAM Cleanup...")
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()

        video_output = vae.decode(samples["samples"])
        if len(video_output.shape) == 5 and video_output.shape[0] == 1: 
            video_output = video_output.squeeze(0)
        
        last_frame = video_output[-1:].clone()
        
        return (video_output, last_frame, None, clip, vae, cv)

# Registered via __init__.py
