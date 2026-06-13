# -*- coding: utf-8 -*-
import logging
import torch
from .domain.models import GenerationSettings, VideoStoryContext
from .application.video_service import VideoGenerationService
from .infrastructure.image_io_adapter import ImageIOAdapter
from .infrastructure.error_adapter import hex_error_handler

logger = logging.getLogger(__name__)

class WanStorySamplerHex:
    """
    [HEX] v3.1.0 Integrated Story Sampler for Wan 2.2.
    Visual headers restored in correct positions.
    """
    @classmethod
    def INPUT_TYPES(cls):
        styles_map = ImageIOAdapter.load_styles_csv()
        style_names = list(styles_map.keys())
        
        return {
            "required": {
                # --- SECTION: INPUTS ---
                "section_story": ("STRING", {"default": "STORY SETTINGS"}),
                "reference_image": ("IMAGE",), 
                "positive_text": ("STRING", {"multiline": True, "forceInput": True}),
                "style_1": (style_names, {"default": "No Style"}),
                "style_2": (style_names, {"default": "No Style"}),
                
                # --- SECTION: MODELS ---
                "section_models": ("STRING", {"default": "GENERATION MODELS"}),
                "model_high": ("MODEL",),
                "model_low": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "clip_vision": ("CLIP_VISION",),
                
                # --- SECTION: PARAMETERS ---
                "section_sampling": ("STRING", {"default": "SAMPLING PARAMETERS"}),
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
    FUNCTION = "execute"
    CATEGORY = "flux_collection_advanced/hex"
    DESCRIPTION = (
        "[HEX] v3.1.0 Integrated Story Sampler for Wan 2.2.\nVisual headers restored in correct positions.\n\n"
        "This documentation was AI-generated. If you find any errors or have suggestions for "
        "improvement, please feel free to contribute! Edit on GitHub."
    )

    @hex_error_handler
    def execute(self, **kwargs):
        settings = GenerationSettings(
            width=kwargs.get("width", 480),
            height=kwargs.get("height", 848),
            num_frames=kwargs.get("num_frames", 81),
            steps_high=kwargs.get("steps_high", 4),
            steps_low=kwargs.get("steps_low", 4),
            cfg=kwargs.get("cfg", 1.0),
            seed=kwargs.get("seed", 0),
            denoise=kwargs.get("denoise", 1.0),
            upscale_method=kwargs.get("upscale_method", "nearest-exact"),
            crop_position=kwargs.get("crop_position", "center")
        )
        context = VideoStoryContext(
            positive_prompt=kwargs.get("positive_text", ""),
            negative_prompt="blurry, low quality, distorted, static, oversaturated",
            style_1=kwargs.get("style_1", "No Style"),
            style_2=kwargs.get("style_2", "No Style")
        )
        models = {
            "model_high": kwargs.get("model_high"),
            "model_low": kwargs.get("model_low"),
            "clip": kwargs.get("clip"),
            "vae": kwargs.get("vae"),
            "clip_vision": kwargs.get("clip_vision")
        }
        styles_map = ImageIOAdapter.load_styles_csv()
        service = VideoGenerationService(style_map=styles_map)
        video_out, samples = service.generate_story_segment(settings, context, models, kwargs.get("reference_image"))
        return (video_out, video_out[-1:].clone())
