# -*- coding: utf-8 -*-
import os
import logging
import torch
import folder_paths
import comfy.sd
import comfy.utils
import comfy.clip_vision
from typing import Any, Dict, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WanVideoLoader_Dev:
    """
    [DEV] v1.7.0 Lightweight Component Loader for Wan 2.2.
    Focuses on CLIP, VAE, and Vision components.
    Models and LoRAs should be handled by original nodes for maximum compatibility.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        clip_list = sorted(folder_paths.get_filename_list("text_encoders") + folder_paths.get_filename_list("clip"))
        vae_list = sorted(folder_paths.get_filename_list("vae"))
        cv_list = sorted(folder_paths.get_filename_list("clip_vision"))

        return {
            "required": {
                "clip_name": (clip_list,),
                "vae_name": (vae_list,),
                "clip_vision_name": (cv_list,),
            }
        }

    RETURN_TYPES = ("CLIP", "VAE", "CLIP_VISION",)
    RETURN_NAMES = ("CLIP", "VAE", "CLIP_VISION",)
    FUNCTION = "load_components"
    CATEGORY = "flux_collection_advanced/_dev"

    def load_components(self, clip_name, vae_name, clip_vision_name):
        logger.info(f"[DEV] Loading Wan Video Components (CLIP, VAE, Vision)")

        # 1. Load CLIP (WAN Type)
        clip_path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
        try: clip_type = comfy.sd.CLIPType.WAN
        except: clip_type = comfy.sd.CLIPType.SD3
        clip = comfy.sd.load_clip(ckpt_paths=[clip_path], clip_type=clip_type)

        # 2. Load VAE
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))

        # 3. Load CLIP Vision
        cv_path = folder_paths.get_full_path_or_raise("clip_vision", clip_vision_name)
        cv = comfy.clip_vision.load(cv_path)

        return (clip, vae, cv)

# Registered via __init__.py
