# -*- coding: utf-8 -*-
import os
import logging
import torch
import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_management
import comfy.clip_vision
from typing import Any, Dict, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WanLoader_Dev:
    """
    [DEV] v1.3.8 Surgical Patch Loader for Wan 2.2.
    Fixed: 'noise_scaling' error by modifying shift in-place without replacing the sampling object.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        unet_list = sorted(folder_paths.get_filename_list("diffusion_models") + folder_paths.get_filename_list("unet"))
        clip_list = sorted(folder_paths.get_filename_list("text_encoders") + folder_paths.get_filename_list("clip"))
        vae_list = sorted(folder_paths.get_filename_list("vae"))
        clip_vision_list = sorted(folder_paths.get_filename_list("clip_vision"))
        lora_list = ["None"] + sorted(folder_paths.get_filename_list("loras"))

        return {
            "required": {
                "model_high": (unet_list,),
                "model_low": (unet_list,),
                "clip_name": (clip_list,),
                "vae_name": (vae_list,),
                "clip_vision_name": (clip_vision_list,),
                "lora_name": (lora_list,),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "sampling_shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("MODEL", "MODEL", "CLIP", "VAE", "CLIP_VISION",)
    RETURN_NAMES = ("MODEL_HIGH", "MODEL_LOW", "CLIP", "VAE", "CLIP_VISION",)
    FUNCTION = "load"
    CATEGORY = "flux_collection_advanced/_dev"

    def _load_model_smart(self, name):
        if name.lower().endswith(".gguf"):
            from nodes import NODE_CLASS_MAPPINGS
            return NODE_CLASS_MAPPINGS["UnetLoaderGGUF"]().load_unet(name)[0]
        
        path = folder_paths.get_full_path("diffusion_models", name) or folder_paths.get_full_path("unet", name)
        return comfy.sd.load_diffusion_model(path)

    def load(self, model_high, model_low, clip_name, vae_name, clip_vision_name, lora_name, lora_strength, sampling_shift):
        logger.info(f"[DEV] Modular Loader v1.3.8: Surgical Patching")

        # 1. Load Core Models
        m_high = self._load_model_smart(model_high)
        m_low = self._load_model_smart(model_low)

        # 2. Load CLIP
        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        try: clip_type = comfy.sd.CLIPType.WAN
        except: clip_type = comfy.sd.CLIPType.SD3
        clip = comfy.sd.load_clip(ckpt_paths=[clip_path], clip_type=clip_type)

        # 3. Load VAE & Vision
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", vae_name)))
        cv = comfy.clip_vision.load(folder_paths.get_full_path_or_raise("clip_vision", clip_vision_name))

        # 4. LoRA
        if lora_name != "None":
            logger.info(f"[DEV] Applying LoRA: {lora_name}")
            lora_path = folder_paths.get_full_path("loras", lora_name)
            lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
            clean_lora = {k.replace("diffusion_model.", ""): v for k, v in lora_sd.items()}
            m_high, clip = comfy.sd.load_lora_for_models(m_high, clip, clean_lora, lora_strength, lora_strength)
            m_low, _ = comfy.sd.load_lora_for_models(m_low, None, clean_lora, lora_strength, 0.0)

        # 5. SURGICAL SHIFT PATCHING (No Object Replacement)
        # We clone the model and modify the internal sampling object's attribute directly.
        # This keeps the original class (with noise_scaling) intact.
        logger.info(f"[DEV] Modifying Sampling Shift to {sampling_shift} (In-Place)")
        
        for i, m in enumerate([m_high, m_low]):
            # Clone model to avoid affecting global state
            patched_m = m.clone()
            
            # Access the underlying model sampling object
            # We assume it has a set_parameters or we set the attribute directly if possible
            sampling_obj = patched_m.model.model_sampling
            
            if hasattr(sampling_obj, "set_parameters"):
                # Safe method
                # We need to ensure we don't change the global object if it's shared
                # So we clone the sampling object first
                import copy
                new_sampling = copy.deepcopy(sampling_obj)
                new_sampling.set_parameters(shift=sampling_shift)
                patched_m.add_object_patch("model_sampling", new_sampling)
            
            if i == 0: m_high = patched_m
            else: m_low = patched_m

        return (m_high, m_low, clip, vae, cv)

# Registered via __init__.py
