# Task-Source: T#27
# -*- coding: utf-8 -*-
import logging
try:
    from infrastructure.doc_adapter import hex_node_doc
except ImportError:
    from .infrastructure.doc_adapter import hex_node_doc

import torch
import folder_paths
import comfy.utils
import comfy.sd
import comfy.samplers
import comfy.model_management
import nodes
from .infrastructure.error_adapter import hex_error_handler

logger = logging.getLogger(__name__)


@hex_node_doc
class FluxLoraDetailerHex:
    """[HEX] LoRA Detailer — aplica un refinamiento con LoRA a una imagen escalada (Img2Img) usando optimizaciones como VAE Tiling."""

    FUNCTION = "execute"
    CATEGORY = "flux_collection_advanced/hex"
    RETURN_TYPES = ("IMAGE", "LATENT",)
    RETURN_NAMES = ("image", "latent",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "section_detailer": ("STRING", {"default": "LORA DETAILER"}),
                "model": ("MODEL", {"tooltip": "Base Flux model."}),
                "clip": ("CLIP", {"tooltip": "Text encoder for prompt encoding."}),
                "vae": ("VAE", {"tooltip": "Used for Encode/Decode cycles."}),
                "image": ("IMAGE", {"tooltip": "Input image (usually from an Upscaler) to be refined."}),
                "lora_name": (lora_list, {"tooltip": "Select the LoRA to add details."}),
                "strength_lora": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "Overall weight of the LoRA."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "Weight applied to the UNET model."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "Weight applied to the CLIP encoder."}),
                "denoise": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Lower values preserve the original image."}),
                "positive_prompt": ("STRING", {"multiline": True, "default": "photorealistic, high detail, sharp focus, skin texture", "tooltip": "Detail-oriented positive prompt."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "blurry, low quality, distorted, cartoon", "tooltip": "Quality-oriented negative prompt."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "Refinement steps."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "CFG Scale (keep at 1.0 for Flux)."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "Sampling method."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "Scheduler type."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed for detail variance."}),
                "vae_tiling": (["enabled", "disabled"], {"default": "enabled", "tooltip": "Essential for 2K/4K upscales. Processes the large image in small tiles."}),
            }
        }

    @hex_error_handler
    def execute(self, **kwargs):
        model = kwargs["model"]
        clip = kwargs["clip"]
        vae = kwargs["vae"]
        image = kwargs["image"]
        lora_name = kwargs["lora_name"]
        strength_lora = kwargs["strength_lora"]
        strength_model = kwargs["strength_model"]
        strength_clip = kwargs["strength_clip"]
        denoise = kwargs["denoise"]
        positive_prompt = kwargs["positive_prompt"]
        negative_prompt = kwargs["negative_prompt"]
        steps = kwargs["steps"]
        cfg = kwargs["cfg"]
        sampler_name = kwargs["sampler_name"]
        scheduler = kwargs["scheduler"]
        seed = kwargs["seed"]
        vae_tiling = kwargs["vae_tiling"]

        logger.info(f"[HEX] Executing LoRA Detailer with Tiling ({vae_tiling}): {lora_name}")

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(
            model, clip, lora, strength_model * strength_lora, strength_clip * strength_lora
        )

        tokens_pos = clip_lora.tokenize(positive_prompt)
        cond_pos, pooled_pos = clip_lora.encode_from_tokens(tokens_pos, return_pooled=True)
        conditioning_pos = [[cond_pos, {"pooled_output": pooled_pos}]]

        tokens_neg = clip_lora.tokenize(negative_prompt)
        cond_neg, pooled_neg = clip_lora.encode_from_tokens(tokens_neg, return_pooled=True)
        conditioning_neg = [[cond_neg, {"pooled_output": pooled_neg}]]

        encoded_pixels = vae.encode(image[:, :, :, :3])
        latents = {"samples": encoded_pixels}

        try:
            samples_dict = nodes.common_ksampler(
                model_lora, seed, steps, cfg, sampler_name, scheduler,
                conditioning_pos, conditioning_neg, latents, denoise=denoise
            )[0]
            sampled_tensor = samples_dict["samples"]
        except Exception as e:
            logger.error(f"[HEX] Sampling failed in Detailer: {e}")
            raise RuntimeError(f"Error during detailer sampling: {e}")

        comfy.model_management.soft_empty_cache()

        logger.info("[HEX] Starting optimized VAE decoding...")
        if vae_tiling == "enabled":
            result_image = vae.decode_tiled(sampled_tensor)
        else:
            result_image = vae.decode(sampled_tensor)

        logger.info("[HEX] LoRA Detailer refinement completed successfully.")
        return (result_image, {"samples": sampled_tensor})
