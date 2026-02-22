# -*- coding: utf-8 -*-
import os
import logging
import torch
import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_sampling
import comfy.clip_vision
from typing import Any, Dict, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WanVideoLoader_Dev:
    """
    [DEV] v1.3.3 Ultimate Loader for Wan 2.2.
    Fixed: Advanced LoRA remapping to solve 'borroso/saturado' issues.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        unet_list = sorted(folder_paths.get_filename_list("diffusion_models") + folder_paths.get_filename_list("unet"))
        clip_list = sorted(folder_paths.get_filename_list("text_encoders") + folder_paths.get_filename_list("clip"))
        vae_list = sorted(folder_paths.get_filename_list("vae"))
        lora_list = ["None"] + sorted(folder_paths.get_filename_list("loras"))
        clip_vision_list = sorted(folder_paths.get_filename_list("clip_vision"))

        return {
            "required": {
                "model_high_noise": (unet_list, {"tooltip": "Select Wan 2.2 High Noise model."}),
                "model_low_noise": (unet_list, {"tooltip": "Select Wan 2.2 Low Noise model."}),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "bf16"], {"default": "default"}),
                "clip_name": (clip_list, {"tooltip": "Select UM T5 / WAN encoder."}),
                "clip_vision_name": (clip_vision_list, {"tooltip": "Select CLIP Vision model."}),
                "vae_name": (vae_list, {"tooltip": "Select Wan Video VAE."}),
                "sampling_shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "t5_optimization": (["None", "Aggressive Offload", "Layer Truncation"], {"default": "Layer Truncation"}),
            },
            "optional": {
                "lora_name": (lora_list, {"default": "None"}),
                "t5_layers": ("INT", {"default": 16, "min": 1, "max": 24, "step": 1}),
            }
        }

    RETURN_TYPES = ("MODEL", "MODEL", "CLIP", "VAE", "CLIP_VISION",)
    RETURN_NAMES = ("MODEL_HIGH", "MODEL_LOW", "CLIP", "VAE", "CLIP_VISION",)
    FUNCTION = "load_wan_video"
    CATEGORY = "flux_collection_advanced/_dev"

    def load_wan_video(self, **kwargs):
        m_high_name = kwargs.get("model_high_noise")
        m_low_name = kwargs.get("model_low_noise")
        weight_dtype = kwargs.get("weight_dtype", "default")
        clip_name = kwargs.get("clip_name")
        cv_name = kwargs.get("clip_vision_name")
        vae_name = kwargs.get("vae_name")
        shift = kwargs.get("sampling_shift", 5.0)
        t5_opt = kwargs.get("t5_optimization", "Layer Truncation")
        lora_name = kwargs.get("lora_name", "None")
        t5_layers = kwargs.get("t5_layers", 16)

        model_options = {}
        if weight_dtype == "fp8_e4m3fn": model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "bf16": model_options["dtype"] = torch.bfloat16

        # 1. Load Models
        p_high = folder_paths.get_full_path("diffusion_models", m_high_name) or folder_paths.get_full_path("unet", m_high_name)
        p_low = folder_paths.get_full_path("diffusion_models", m_low_name) or folder_paths.get_full_path("unet", m_low_name)
        m_high = comfy.sd.load_diffusion_model(p_high, model_options=model_options)
        m_low = comfy.sd.load_diffusion_model(p_low, model_options=model_options)

        # 2. CLIP & Vision
        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        try: clip_type = comfy.sd.CLIPType.WAN
        except: clip_type = comfy.sd.CLIPType.SD3
        clip = comfy.sd.load_clip(ckpt_paths=[clip_path], clip_type=clip_type)
        cv = comfy.clip_vision.load(folder_paths.get_full_path_or_raise("clip_vision", cv_name))

        # 3. LoRA Patching with Robust Remapping
        if lora_name != "None":
            logger.info(f"[DEV] Applying LoRA with Intelligent Remapping: {lora_name}")
            lora_path = folder_paths.get_full_path("loras", lora_name)
            lora_sd = comfy.utils.load_torch_file(lora_path, safe_load=True)
            
            # Map keys: handle both 'diffusion_model.' prefix and direct names
            remap_sd = {}
            for k, v in lora_sd.items():
                new_k = k.replace("diffusion_model.", "")
                remap_sd[new_k] = v
            
            m_high, clip = comfy.sd.load_lora_for_models(m_high, clip, remap_sd, 1.0, 1.0)
            m_low, _ = comfy.sd.load_lora_for_models(m_low, None, remap_sd, 1.0, 0.0)

        # 4. Sampling Shift
        for m in [m_high, m_low]:
            try: m.model.model_sampling.set_parameters(shift=shift)
            except: pass

        # 5. VAE
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", vae_name)))

        return (m_high, m_low, clip, vae, cv)

# Registered via __init__.py
