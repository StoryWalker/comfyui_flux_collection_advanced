# -*- coding: utf-8 -*-
import logging
import folder_paths
from .domain.models import WanModelConfig
from .application.loader_service import ModelLoaderService
from .infrastructure.error_adapter import hex_error_handler

logger = logging.getLogger(__name__)

class WanUnifiedLoaderHex:
    """
    [HEX] v3.1.0 Unified Wan Loader.
    Restored visual distribution with validation-safe headers.
    """
    @classmethod
    def INPUT_TYPES(cls):
        unet_list = sorted(folder_paths.get_filename_list("diffusion_models") + folder_paths.get_filename_list("unet") + folder_paths.get_filename_list("unet_gguf"))
        clip_list = sorted(folder_paths.get_filename_list("text_encoders") + folder_paths.get_filename_list("clip") + folder_paths.get_filename_list("clip_gguf"))
        vae_list = sorted(folder_paths.get_filename_list("vae"))
        cv_list = sorted(folder_paths.get_filename_list("clip_vision"))
        lora_list = ["None"] + sorted(folder_paths.get_filename_list("loras"))

        return {
            "required": {
                # --- SECTION: UNET MODELS ---
                "section_unet": ("STRING", {"default": "UNET CONFIGURATION"}),
                "model_high": (unet_list,),
                "model_low": (unet_list,),
                "weight_dtype": (["default", "fp8_e4m3fn", "bf16"], {"default": "default"}),
                "sampling_shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                
                # --- SECTION: ENCODERS ---
                "section_encoders": ("STRING", {"default": "TEXT & VISION ENCODERS"}),
                "clip_name": (clip_list,),
                "t5_optimization": (["None", "Layer Truncation", "Aggressive Offload"], {"default": "Layer Truncation"}),
                "t5_layers": ("INT", {"default": 16, "min": 1, "max": 24, "step": 1}),
                "clip_vision_name": (cv_list,),
                "vae_name": (vae_list,),
                
                # --- SECTION: LORA PATCHING ---
                "section_lora": ("STRING", {"default": "LORA ADAPTERS"}),
                "lora_name": (lora_list, {"default": "None"}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "MODEL", "CLIP", "VAE", "CLIP_VISION",)
    RETURN_NAMES = ("MODEL_HIGH", "MODEL_LOW", "CLIP", "VAE", "CLIP_VISION",)
    FUNCTION = "execute"
    CATEGORY = "flux_collection_advanced/hex"
    DESCRIPTION = (
        "[HEX] v3.1.0 Unified Wan Loader.\nRestored visual distribution with validation-safe headers.\n\n"
        "This documentation was AI-generated. If you find any errors or have suggestions for "
        "improvement, please feel free to contribute! Edit on GitHub."
    )

    @hex_error_handler
    def execute(self, **kwargs):
        config = WanModelConfig(
            model_high_name=kwargs.get("model_high"),
            model_low_name=kwargs.get("model_low"),
            clip_name=kwargs.get("clip_name"),
            vae_name=kwargs.get("vae_name"),
            clip_vision_name=kwargs.get("clip_vision_name"),
            sampling_shift=kwargs.get("sampling_shift", 5.0),
            weight_dtype=kwargs.get("weight_dtype", "default"),
            lora_name=kwargs.get("lora_name", "None"),
            lora_strength=kwargs.get("lora_strength", 1.0),
            t5_optimization=kwargs.get("t5_optimization", "Layer Truncation"),
            t5_layers=kwargs.get("t5_layers", 16)
        )
        service = ModelLoaderService()
        stack = service.load_full_wan_stack(config)
        return (stack["model_high"], stack["model_low"], stack["clip"], stack["vae"], stack["clip_vision"])
