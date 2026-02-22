# -*- coding: utf-8 -*-
import torch
import logging
import nodes
import comfy.model_management
from typing import Any, Dict, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WanDualSampler_Dev:
    """
    [DEV] Modular Dual-Stage Sampler for Wan 2.2.
    Focuses only on sequential sampling with noise handover.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_high": ("MODEL",),
                "model_low": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "steps_high": ("INT", {"default": 4, "min": 1, "max": 50}),
                "steps_low": ("INT", {"default": 4, "min": 1, "max": 50}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "flux_collection_advanced/_dev"

    def sample(self, model_high, model_low, positive, negative, latent_image, 
               steps_high, steps_low, cfg, seed, denoise):
        
        total_steps = steps_high + steps_low
        
        # 1. Stage 1: High Noise
        logger.info(f"[DEV] Modular Sampler Stage 1: {steps_high} steps")
        samples = nodes.common_ksampler(
            model_high, seed, total_steps, cfg, "lcm", "simple", 
            positive, negative, latent_image, 
            denoise=denoise, disable_noise=False, start_step=0, last_step=steps_high, 
            force_full_denoise=False
        )[0]

        # 2. Stage 2: Low Noise
        logger.info(f"[DEV] Modular Sampler Stage 2: {steps_low} steps")
        comfy.model_management.soft_empty_cache()
        
        samples = nodes.common_ksampler(
            model_low, seed, total_steps, cfg, "lcm", "simple", 
            positive, negative, samples, 
            denoise=denoise, disable_noise=True, # Resume from previous noise
            start_step=steps_high, last_step=total_steps, 
            force_full_denoise=True
        )[0]

        return (samples,)

# Registered via __init__.py
