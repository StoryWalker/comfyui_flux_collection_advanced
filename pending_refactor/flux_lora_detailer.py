# -*- coding: utf-8 -*-
import logging
import torch
import folder_paths
import comfy.utils
import comfy.sd
import comfy.model_management
import nodes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FluxLoraDetailer:
    """
    Advanced Refinement Node:
    Applies LoRA detailing to an upscaled image (Img2Img) with VRAM optimizations 
    like VAE Tiling and memory flushing to support high-resolution (2K/4K) outputs.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Base Flux model."}),
                "clip": ("CLIP", {"tooltip": "Text encoder for prompt encoding."}),
                "vae": ("VAE", {"tooltip": "Used for Encode/Decode cycles."}),
                "image": ("IMAGE", {"tooltip": "Input image (usually from an Upscaler) to be refined."}),
                "lora_name": (lora_list, {"tooltip": "Select the LoRA to add details (e.g., FluxMythR3alistic)."}),
                "strength_lora": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "Overall weight of the LoRA."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "Weight applied to the UNET model."}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, "tooltip": "Weight applied to the CLIP encoder."}),
                "denoise": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Lower values (0.2-0.3) preserve the original upscaled image while adding texture."}),
                "positive_prompt": ("STRING", {"multiline": True, "default": "photorealistic, high detail, sharp focus, skin texture", "tooltip": "Detail-oriented positive prompt."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "blurry, low quality, distorted, cartoon", "tooltip": "Quality-oriented negative prompt."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "Refinement steps (10-20 is usually enough)."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "CFG Scale (keep at 1.0 for Flux)."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "Sampling method."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "Scheduler type."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed for detail variance."}),
                "vae_tiling": (["enabled", "disabled"], {"default": "enabled", "tooltip": "Essential for 2K/4K upscales. Processes the large image in small tiles."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT",)
    RETURN_NAMES = ("image", "latent",)
    FUNCTION = "apply_detail"
    CATEGORY = "flux_collection_advanced"

    def apply_detail(self, model, clip, vae, image, lora_name, strength_lora, strength_model, strength_clip, 
                     denoise, positive_prompt, negative_prompt, steps, cfg, sampler_name, scheduler, seed, vae_tiling):
        
        logger.info(f"Executing LoRA Detailer with Tiling ({vae_tiling}): {lora_name}")

        # 1. Load and Apply LoRA
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        
        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model * strength_lora, strength_clip * strength_lora)
        
        # 2. Prepare Conditioning (Prompting)
        tokens_pos = clip_lora.tokenize(positive_prompt)
        cond_pos, pooled_pos = clip_lora.encode_from_tokens(tokens_pos, return_pooled=True)
        conditioning_pos = [[cond_pos, {"pooled_output": pooled_pos}]]

        tokens_neg = clip_lora.tokenize(negative_prompt)
        cond_neg, pooled_neg = clip_lora.encode_from_tokens(tokens_neg, return_pooled=True)
        conditioning_neg = [[cond_neg, {"pooled_output": pooled_neg}]]

        # 3. Encode Image to Latent (with memory check)
        # Handle alpha/masking if present by slicing to RGB
        encoded_pixels = vae.encode(image[:,:,:,:3])
        latents = {"samples": encoded_pixels}
        
        # 4. Sampling (Refinement Pass)
        try:
            samples_dict = nodes.common_ksampler(
                model_lora, seed, steps, cfg, sampler_name, scheduler, 
                conditioning_pos, conditioning_neg, latents, denoise=denoise
            )[0]
            sampled_tensor = samples_dict["samples"]
        except Exception as e:
            logger.error(f"Sampling failed in Detailer: {e}")
            raise RuntimeError(f"Error during detailer sampling: {e}")

        # 5. Decode Latent to Image (Optimized)
        # Free memory from the sampling model before VAE decoding
        comfy.model_management.soft_empty_cache()

        logger.info("Starting optimized VAE decoding...")
        if vae_tiling == "enabled":
            # Decodes in tiles to support high-res upscales (2K/4K)
            result_image = vae.decode_tiled(sampled_tensor)
        else:
            result_image = vae.decode(sampled_tensor)
        
        logger.info("LoRA Detailer refinement completed successfully.")
        return (result_image, {"samples": sampled_tensor})

# Registration is handled in __init__.py
