# -*- coding: utf-8 -*-
import logging
import torch
import folder_paths
import comfy.sd
import comfy.utils
from typing import Any, Dict, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WanVideoLoader_Dev:
    """
    [DEV] Specialized Dual Loader for Wan 2.2.
    Loads both High Noise and Low Noise models, providing separate outputs
    to allow sequential sampling without keeping both in VRAM simultaneously.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        unet_list = sorted(folder_paths.get_filename_list("diffusion_models") + folder_paths.get_filename_list("unet"))
        clip_list = sorted(folder_paths.get_filename_list("text_encoders") + folder_paths.get_filename_list("clip"))
        vae_list = sorted(folder_paths.get_filename_list("vae"))

        return {
            "required": {
                "model_high_noise": (unet_list, {"tooltip": "Select Wan 2.2 High Noise model (Initial structure)."}),
                "model_low_noise": (unet_list, {"tooltip": "Select Wan 2.2 Low Noise model (Detail refiner)."}),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "bf16"], {"default": "default"}),
                "clip_name": (clip_list, {"tooltip": "Select UM T5 or compatible encoder."}),
                "vae_name": (vae_list, {"tooltip": "Select Wan Video VAE."}),
                "t5_optimization": (["None", "Aggressive Offload", "Layer Truncation"], {"default": "None"}),
            },
            "optional": {
                "t5_layers": ("INT", {"default": 24, "min": 1, "max": 24, "step": 1}),
            }
        }

    RETURN_TYPES = ("MODEL", "MODEL", "CLIP", "VAE",)
    RETURN_NAMES = ("MODEL_HIGH", "MODEL_LOW", "CLIP", "VAE",)
    FUNCTION = "load_wan_video"
    CATEGORY = "flux_collection_advanced/_dev"

    def load_wan_video(self, model_high_noise, model_low_noise, weight_dtype, clip_name, vae_name, t5_optimization, t5_layers=24):
        # Asegurarse de que t5_layers sea un entero, manejando posibles desplazamientos de widgets
        if not isinstance(t5_layers, int):
            try:
                t5_layers = int(t5_layers)
            except:
                t5_layers = 24
        
        logger.info(f"[DEV] Loading Wan 2.2 Dual Architecture | T5 Layers: {t5_layers}")

        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "bf16":
            model_options["dtype"] = torch.bfloat16

        # 1. Load Models
        models = []
        for m_name in [model_high_noise, model_low_noise]:
            path = folder_paths.get_full_path("diffusion_models", m_name) or \
                   folder_paths.get_full_path("unet", m_name)
            if not path:
                raise FileNotFoundError(f"Model {m_name} not found.")
            
            logger.info(f"[DEV] Loading component: {m_name}")
            models.append(comfy.sd.load_diffusion_model(path, model_options=model_options))

        # 2. Load Shared CLIP (T5)
        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        # Fix: Use CLIPType.WAN for the UM T5 encoder required by Wan 2.1/2.2
        try:
            clip_type_enum = comfy.sd.CLIPType.WAN
        except AttributeError:
            # Fallback if WAN is not available (though it should be based on discovery)
            clip_type_enum = comfy.sd.CLIPType.SD3
            
        clip = comfy.sd.load_clip(ckpt_paths=[clip_path], clip_type=clip_type_enum)

        # 3. Apply T5 Optimizations
        if t5_optimization == "Layer Truncation":
            try:
                t5 = None
                if hasattr(clip.cond_stage_model, "t5xxl"): t5 = clip.cond_stage_model.t5xxl.transformer
                elif hasattr(clip.cond_stage_model, "transformer"): t5 = clip.cond_stage_model.transformer
                if t5 and hasattr(t5, "encoder") and t5_layers < 24:
                    t5.encoder.block = t5.encoder.block[:t5_layers]
            except: pass

        # 4. Load shared Video VAE
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))

        return (models[0], models[1], clip, vae)

# Registered via __init__.py
