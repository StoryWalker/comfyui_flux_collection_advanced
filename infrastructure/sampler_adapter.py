# -*- coding: utf-8 -*-
import torch
import logging
import nodes
import comfy.model_management
from typing import Any, Dict

logger = logging.getLogger(__name__)

class ComfySamplerAdapter:
    """
    Infrastructure Adapter for ComfyUI Sampling Engine.
    Manages VRAM and sequential High/Low noise passes.
    """
    
    @staticmethod
    def run_dual_stage_lcm(model_high: Any, model_low: Any, seed: int, steps_high: int, steps_low: int, 
                           cfg: float, positive: Any, negative: Any, latent_dict: dict, denoise: float) -> Any:
        
        total_steps = steps_high + steps_low
        
        # Phase 1: Structural (High Noise)
        logger.info(f"[HEX] Sampling Stage 1 (High): 0 to {steps_high}")
        samples = nodes.common_ksampler(
            model_high, seed, total_steps, cfg, "lcm", "simple", 
            positive, negative, latent_dict, 
            denoise=denoise, disable_noise=False, start_step=0, last_step=steps_high, 
            force_full_denoise=False
        )[0]

        # VRAM Management
        comfy.model_management.soft_empty_cache()

        # Phase 2: Textural (Low Noise)
        logger.info(f"[HEX] Sampling Stage 2 (Low): {steps_high} to {total_steps}")
        samples = nodes.common_ksampler(
            model_low, seed, total_steps, cfg, "lcm", "simple", 
            positive, negative, samples, 
            denoise=denoise, disable_noise=True, 
            start_step=steps_high, last_step=total_steps, 
            force_full_denoise=True
        )[0]

        return samples

    @staticmethod
    def final_cleanup():
        """ Forces VRAM purge to prevent black screens in chains """
        logger.info("[HEX] Final VRAM Flush")
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
