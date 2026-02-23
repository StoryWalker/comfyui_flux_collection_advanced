# -*- coding: utf-8 -*-
import logging
import folder_paths
from .infrastructure.model_adapter import ComfyModelAdapter
from .infrastructure.error_adapter import hex_error_handler

logger = logging.getLogger(__name__)

class Test_Hex_UnetLoader:
    @classmethod
    def INPUT_TYPES(cls):
        unet_list = sorted(folder_paths.get_filename_list("diffusion_models") + folder_paths.get_filename_list("unet") + folder_paths.get_filename_list("unet_gguf"))
        return {
            "required": {
                "unet_name": (unet_list,),
                "weight_dtype": (["default", "fp8_e4m3fn", "bf16"], {"default": "default"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "execute"
    CATEGORY = "flux_collection_advanced/test"

    @hex_error_handler
    def execute(self, unet_name, weight_dtype):
        adapter = ComfyModelAdapter()
        model = adapter.load_unet(unet_name, weight_dtype)
        return (model,)

class Test_Hex_ClipLoader:
    @classmethod
    def INPUT_TYPES(cls):
        clip_list = sorted(folder_paths.get_filename_list("text_encoders") + folder_paths.get_filename_list("clip") + folder_paths.get_filename_list("clip_gguf"))
        return {
            "required": {
                "clip_name": (clip_list,),
                "t5_optimization": (["None", "Layer Truncation", "Aggressive Offload"], {"default": "Layer Truncation"}),
                "t5_layers": ("INT", {"default": 16, "min": 1, "max": 24, "step": 1}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "execute"
    CATEGORY = "flux_collection_advanced/test"

    @hex_error_handler
    def execute(self, clip_name, t5_optimization, t5_layers):
        adapter = ComfyModelAdapter()
        clip = adapter.load_clip_wan(clip_name, t5_optimization, t5_layers)
        return (clip,)

class Test_Hex_LoraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        lora_list = ["None"] + sorted(folder_paths.get_filename_list("loras"))
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_name": (lora_list,),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP",)
    FUNCTION = "execute"
    CATEGORY = "flux_collection_advanced/test"

    @hex_error_handler
    def execute(self, model, clip, lora_name, strength_model, strength_clip):
        if lora_name == "None":
            return (model, clip)
        adapter = ComfyModelAdapter()
        m, c = adapter.apply_lora_robust(model, clip, lora_name, strength_model)
        return (m, c)

class Test_Hex_ModelSampling:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 20.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "execute"
    CATEGORY = "flux_collection_advanced/test"

    @hex_error_handler
    def execute(self, model, shift):
        adapter = ComfyModelAdapter()
        m = adapter.apply_sampling_shift(model, shift)
        return (m,)

class Test_Hex_VaeLoader:
    @classmethod
    def INPUT_TYPES(cls):
        vae_list = sorted(folder_paths.get_filename_list("vae"))
        return {
            "required": {
                "vae_name": (vae_list,),
            }
        }

    RETURN_TYPES = ("VAE",)
    FUNCTION = "execute"
    CATEGORY = "flux_collection_advanced/test"

    @hex_error_handler
    def execute(self, vae_name):
        adapter = ComfyModelAdapter()
        vae, _ = adapter.load_vae_and_vision(vae_name, "")
        return (vae,)

class Test_Hex_ClipVisionLoader:
    @classmethod
    def INPUT_TYPES(cls):
        cv_list = sorted(folder_paths.get_filename_list("clip_vision"))
        return {
            "required": {
                "clip_vision_name": (cv_list,),
            }
        }

    RETURN_TYPES = ("CLIP_VISION",)
    FUNCTION = "execute"
    CATEGORY = "flux_collection_advanced/test"

    @hex_error_handler
    def execute(self, clip_vision_name):
        adapter = ComfyModelAdapter()
        _, cv = adapter.load_vae_and_vision("", clip_vision_name)
        return (cv,)
