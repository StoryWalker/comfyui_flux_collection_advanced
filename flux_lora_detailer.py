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
    A node that applies a LoRA to a Flux model and then uses it to add details 
    to an existing image (img2img) through a secondary sampling pass.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras")
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "lora_name": (lora_list, {"tooltip": "Select the LoRA to add details (e.g., FluxMythR3alistic)."}),
                "strength_lora": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "denoise": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Lower values preserve more of the original image."}),
                "positive_prompt": ("STRING", {"multiline": True, "default": "photorealistic, high detail, sharp focus"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "blurry, low quality, distorted"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_detail"
    CATEGORY = "flux_collection_advanced"

    def apply_detail(self, model, clip, vae, image, lora_name, strength_lora, strength_model, strength_clip, 
                     denoise, positive_prompt, negative_prompt, steps, cfg, sampler_name, scheduler, seed):
        
        logger.info(f"Applying LoRA detailer: {lora_name} with denoise {denoise}")

        # 1. Load and Apply LoRA
        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        
        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model * strength_lora, strength_clip * strength_lora)
        
        # 2. Prepare Conditioning (Prompting)
        # Using standard CLIP encoding
        tokens_pos = clip_lora.tokenize(positive_prompt)
        cond_pos, pooled_pos = clip_lora.encode_from_tokens(tokens_pos, return_pooled=True)
        conditioning_pos = [[cond_pos, {"pooled_output": pooled_pos}]]

        tokens_neg = clip_lora.tokenize(negative_prompt)
        cond_neg, pooled_neg = clip_lora.encode_from_tokens(tokens_neg, return_pooled=True)
        conditioning_neg = [[cond_neg, {"pooled_output": pooled_neg}]]

        # 3. Encode Image to Latent
        # ComfyUI images are [B, H, W, C], VAE expects [B, C, H, W] handled by vae.encode
        # The sampler expects a dictionary with a "samples" key.
        encoded_pixels = vae.encode(image[:,:,:,:3])
        latents = {"samples": encoded_pixels}
        
        # 4. Sampling (img2img)
        # We use a lower denoise to keep the structure of the upscaled image
        # Calculate steps to run based on denoise
        total_steps = steps
        start_step = int(total_steps * (1.0 - denoise))
        
        # Execute Sampling
        disable_noise = False
        if denoise >= 1.0:
            disable_noise = False
        
        # Using KSampler logic
        try:
            samples = nodes.common_ksampler(
                model_lora, seed, total_steps, cfg, sampler_name, scheduler, 
                conditioning_pos, conditioning_neg, latents, denoise=denoise
            )[0]
        except Exception as e:
            logger.error(f"Sampling failed: {e}")
            raise RuntimeError(f"Error during detailer sampling: {e}")

        # 5. Decode Latent to Image
        # samples is a dictionary containing the "samples" tensor
        result_image = vae.decode(samples["samples"])
        
        logger.info("LoRA Detailer pass completed.")
        return (result_image,)

# Registration is handled in __init__.py
