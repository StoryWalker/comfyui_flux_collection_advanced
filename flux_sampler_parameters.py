import logging
from typing import Any, Dict, Tuple, Optional, Callable

# Necessary third-party imports
import torch
import comfy.model_management
import comfy.samplers
import comfy.sample
import comfy.utils
import latent_preview
import nodes # For inheritance and MAX_RESOLUTION

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FluxSamplerParameters(nodes.ComfyNodeABC):
    """
    Advanced Flux sampler that auto-detects model architecture (channels and text dimensions).
    Automatically handles padding for conditioning mismatches (e.g., 4096 to 6144)
    to support GGUF and distilled variants.
    """

    # --- Node Metadata for ComfyUI ---
    FUNCTION = "generate_sample_and_decode"
    CATEGORY = "flux_collection_advanced"
    DESCRIPTION = "Advanced Flux Sampler with auto-arch detection and conditioning padding."
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """ Define the required input types for the node. """
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "vae": ("VAE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),                
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    # --- Core Logic Methods ---

    def _get_device(self) -> torch.device:
        return comfy.model_management.get_torch_device()

    def _get_model_info(self, model: Any) -> Dict[str, int]:
        """ Detects image channels and text input features from the model architecture. """
        info = {"channels": 16, "text_dim": 4096}
        try:
            base_model = getattr(model, "model", None)
            diffusion_model = getattr(base_model, "diffusion_model", None)
            
            # 1. Detect Image Channels
            img_in = getattr(diffusion_model, "img_in", None)
            if img_in is not None:
                in_f = getattr(img_in, "in_features", 64)
                if in_f == 64: info["channels"] = 16
                elif in_f == 128: info["channels"] = 128
                else: info["channels"] = 16 # Default fallback
                logger.info(f"Detected image channels: {info['channels']} (img_in: {in_f})")

            # 2. Detect Text Dimension
            txt_in = getattr(diffusion_model, "txt_in", None)
            if txt_in is not None:
                txt_f = getattr(txt_in, "in_features", 4096)
                info["text_dim"] = txt_f
                logger.info(f"Detected model text dimension: {info['text_dim']}")
        except Exception as e:
            logger.warning(f"Metadata detection failed: {e}. Using Flux defaults.")
        return info

    def _prepare_conditioning(self, conditioning: Any, target_dim: int) -> Any:
        """ 
        Ensures conditioning matches the model's expected text dimension.
        Pads with zeros if there is a mismatch (e.g., from 4096 to 6144).
        """
        if not isinstance(conditioning, list): return conditioning
        
        new_conditioning = []
        for item in conditioning:
            t_orig = item[0]
            current_dim = t_orig.shape[-1]
            
            if current_dim != target_dim:
                logger.info(f"Padding conditioning: {current_dim} -> {target_dim}")
                # Pad the last dimension with zeros
                padding = (0, target_dim - current_dim)
                t_padded = torch.nn.functional.pad(t_orig, padding, "constant", 0)
                new_conditioning.append([t_padded, item[1]])
            else:
                new_conditioning.append(item)
        return new_conditioning

    def _generate_empty_latent(self, width: int, height: int, batch_size: int, device: torch.device, channels: int) -> Dict[str, torch.Tensor]:
        compression = 16
        latent = torch.zeros([batch_size, channels, height // compression, width // compression], device=device)
        logger.info(f"Generated Flux latent: {latent.shape} on {device}")
        return {"samples": latent}

    def _sample_latent(self, model: Any, seed: int, steps: int, cfg: float,
                       sampler_name: str, scheduler: str, positive: Any,
                       initial_latent: Dict[str, torch.Tensor], denoise: float
                       ) -> torch.Tensor:
        try:
            latent_image = initial_latent["samples"]
            model_device = model.load_device
            noise = comfy.sample.prepare_noise(latent_image, seed).to(model_device)
            preview_callback = latent_preview.prepare_callback(model, steps)
            
            samples = comfy.sample.sample(
                model=model, noise=noise, steps=steps, cfg=cfg,
                sampler_name=sampler_name, scheduler=scheduler,
                positive=positive, negative='',
                latent_image=latent_image.to(model_device),
                denoise=denoise, disable_noise=False, start_step=None, last_step=None,
                force_full_denoise=False, noise_mask=None,
                callback=preview_callback, disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed
            )
            if samples is None: raise RuntimeError("Sampler returned None.")
            return samples
        except Exception as e:
            logger.exception(f"Sampling error: {e}")
            raise RuntimeError(f"Sampling process failed: {e}") from e

    def _decode_latent(self, vae: Any, latent_samples: torch.Tensor) -> torch.Tensor:
        """ 
        Decodes the sampled latent into an image using the provided VAE.
        Enhanced to handle Nested Tensors, 5D Video Latents, and dynamic channel matching.
        """
        logger.info("Starting VAE decoding process...")
        try:
            # 1. Handle Nested Tensors (Standard ComfyUI practice)
            if hasattr(latent_samples, "is_nested") and latent_samples.is_nested:
                latent_samples = latent_samples.unbind()[0]

            # 2. Handle 5D Video Latents [B, C, T, H, W] -> [B*T, C, H, W]
            # Some advanced models (like Wan 2.1) output temporal dimensions
            original_shape = latent_samples.shape
            is_video = len(original_shape) == 5
            if is_video:
                b, c, t, h, w = original_shape
                # Move temporal dim to batch: [B, T, C, H, W] -> [B*T, C, H, W]
                latent_samples = latent_samples.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
                logger.info(f"Video latent detected. Reshaped from {original_shape} to {latent_samples.shape}")

            # 3. Dynamic Decoding with Channel Fallback
            pixels = None
            try:
                # Attempt decoding with full channels first
                pixels = vae.decode(latent_samples)
            except RuntimeError as e:
                # If it's a channel mismatch error, try smart slicing
                if "channels" in str(e) or "size" in str(e):
                    logger.warning(f"Initial decoding failed ({e}). Attempting channel matching fallback...")
                    
                    # Detect expected channels from VAE architecture
                    vae_channels = 16
                    try:
                        if hasattr(vae, "first_stage_model"):
                            decoder = getattr(vae.first_stage_model, "decoder", None)
                            conv_in = getattr(decoder, "conv_in", None)
                            if conv_in is not None:
                                vae_channels = getattr(conv_in, "in_channels", 16)
                    except: pass
                    
                    current_ch = latent_samples.shape[1]
                    if current_ch > vae_channels:
                        logger.info(f"Slicing latent channels for VAE: {current_ch} -> {vae_channels}")
                        pixels = vae.decode(latent_samples[:, :vae_channels, :, :])
                    else:
                        raise e
                else:
                    raise e

            # 4. Handle 5D Output (Combine batches/frames)
            if pixels is not None and len(pixels.shape) == 5:
                # [B, T, H, W, C] -> [B*T, H, W, C]
                pixels = pixels.reshape(-1, pixels.shape[-3], pixels.shape[-2], pixels.shape[-1])
            
            logger.info(f"Decoding successful. Final image shape: {pixels.shape}")
            return pixels

        except Exception as e:
            logger.exception(f"VAE decoding error: {e}")
            raise RuntimeError(f"Decoding failed: {e}") from e

    # --- Main Execution Function ---
    def generate_sample_and_decode(self, width: int, height: int, batch_size: int,
                                   model: Any, vae: Any, seed: int, steps: int, cfg: float,
                                   sampler_name: str, scheduler: str,
                                   positive: Any, denoise: float = 1.0
                                   ) -> Tuple[torch.Tensor,]:
        node_name = self.__class__.__name__
        logger.info(f"Executing node: {node_name}")
        
        device = self._get_device()
        
        # 1. Detect Architecture
        model_info = self._get_model_info(model)
        
        # 2. Adjust Conditioning (Padding if necessary)
        positive = self._prepare_conditioning(positive, model_info["text_dim"])

        try:
            # 3. Generate Latent
            initial_latent_dict = self._generate_empty_latent(width, height, batch_size, device, model_info["channels"])
            
            # 4. Sample
            sampled_latent = self._sample_latent(
                model=model, seed=seed, steps=steps, cfg=cfg,
                sampler_name=sampler_name, scheduler=scheduler,
                positive=positive,
                initial_latent=initial_latent_dict, denoise=denoise
            )
            
            # 5. Decode
            image = self._decode_latent(vae, sampled_latent)
            return (image,)

        except Exception as e:
             logger.error(f"Execution failed: {e}")
             raise RuntimeError(f"Node execution failed: {e}") from e

# --- Registration ---
# NODE_CLASS_MAPPINGS = { "FluxSamplerParameters": FluxSamplerParameters }

# --- ComfyUI Registration Info ---
# NODE_CLASS_MAPPINGS = { "FluxSamplerParameters": FluxSamplerParameters }
# NODE_DISPLAY_NAME_MAPPINGS = { "FluxSamplerParameters": "Flux Generate, Sample & Decode" }
