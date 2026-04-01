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
    Advanced Flux sampler optimized for Flux.2. 
    Includes:
    1. VAE Tiled Decoding for high-resolution stability.
    2. Optional Latent input for Img2Img/Refinement workflows.
    3. Soft memory management to prevent OOM during handoffs.
    4. Auto-arch detection (channels/text dims).
    """

    # --- Node Metadata for ComfyUI ---
    FUNCTION = "generate_sample_and_decode"
    CATEGORY = "flux_collection_advanced"
    DESCRIPTION = "Advanced Flux Sampler with VAE Tiling (for high-res upscaling) and optional Latent input (for Img2Img/Refinement)."
    RETURN_TYPES = ("IMAGE", "LATENT",)
    RETURN_NAMES = ("image", "latent",)

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """ Define the required and optional input types for the node. """
        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16, "tooltip": "Target image width in pixels."}),
                "height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16, "tooltip": "Target image height in pixels."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "tooltip": "Number of images to generate per run."}),
                "model": ("MODEL", {"tooltip": "Flux model (UNET/Diffusion)."}),
                "positive": ("CONDITIONING", {"tooltip": "Text prompt encoding."}),
                "vae": ("VAE", {"tooltip": "VAE for latent/image conversion."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "Random seed for reproducibility."}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 10000, "tooltip": "Number of sampling steps."}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "Classifier-Free Guidance scale (1.0 is standard for Flux)."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "Choice of sampling algorithm."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "Noise scheduling method."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Strength of noise removal (1.0 for fresh generation, lower for Img2Img)."}),
                "vae_tiling": (["enabled", "disabled"], {"default": "enabled", "tooltip": "Uses tiled VAE decoding (essential for 2K/4K resolutions) to prevent memory errors."}),
            },
            "optional": {
                "latent_opt": ("LATENT", {"tooltip": "Optional latent input for structural refinement or Img2Img mode."}),
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
        logger.info(f"Generated Empty Flux latent: {latent.shape} on {device}")
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

    def _decode_latent(self, vae: Any, latent_samples: torch.Tensor, tiling_mode: str) -> torch.Tensor:
        """ 
        Decodes the sampled latent into an image using the provided VAE.
        Enhanced with Tiling and dynamic channel matching.
        """
        logger.info("Starting VAE decoding process...")
        try:
            # 1. Standardize Latent
            if hasattr(latent_samples, "is_nested") and latent_samples.is_nested:
                latent_samples = latent_samples.unbind()[0]

            # 2. Reshape for Video if needed (5D)
            original_shape = latent_samples.shape
            is_video = len(original_shape) == 5
            if is_video:
                b, c, t, h, w = original_shape
                latent_samples = latent_samples.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)

            # 3. Memory Optimization: Clear cache before VAE heavy lifting
            comfy.model_management.soft_empty_cache()

            # 4. Tiled vs Standard Decoding
            pixels = None
            if tiling_mode == "enabled":
                logger.info("Using Tiled VAE Decoding.")
                pixels = vae.decode_tiled(latent_samples)
            else:
                pixels = vae.decode(latent_samples)

            # 5. Handle Video/Frame combining
            if pixels is not None and len(pixels.shape) == 5:
                pixels = pixels.reshape(-1, pixels.shape[-3], pixels.shape[-2], pixels.shape[-1])
            
            logger.info(f"Decoding successful. Image shape: {pixels.shape}")
            return pixels

        except Exception as e:
            logger.exception(f"VAE decoding error: {e}")
            raise RuntimeError(f"Decoding failed: {e}") from e

    # --- Main Execution Function ---
    def generate_sample_and_decode(self, width: int, height: int, batch_size: int,
                                   model: Any, vae: Any, seed: int, steps: int, cfg: float,
                                   sampler_name: str, scheduler: str,
                                   positive: Any, denoise: float = 1.0, 
                                   vae_tiling: str = "enabled", latent_opt: Optional[Dict[str, Any]] = None
                                   ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        node_name = self.__class__.__name__
        logger.info(f"Executing node: {node_name}")
        
        device = self._get_device()
        
        # 1. Detect Architecture
        model_info = self._get_model_info(model)
        
        # 2. Adjust Conditioning (Padding if necessary)
        positive = self._prepare_conditioning(positive, model_info["text_dim"])

        try:
            # 3. Determine Latent Source (Img2Img support)
            if latent_opt is not None:
                logger.info("Using provided latent input (Img2Img/Refiner mode).")
                initial_latent_dict = latent_opt
            else:
                initial_latent_dict = self._generate_empty_latent(width, height, batch_size, device, model_info["channels"])
            
            # 4. Sample
            sampled_latent = self._sample_latent(
                model=model, seed=seed, steps=steps, cfg=cfg,
                sampler_name=sampler_name, scheduler=scheduler,
                positive=positive,
                initial_latent=initial_latent_dict, denoise=denoise
            )
            
            # Wrap sampled latent for output
            out_latent = {"samples": sampled_latent}

            # 5. Decode
            image = self._decode_latent(vae, sampled_latent, vae_tiling)
            
            return (image, out_latent)

        except Exception as e:
             logger.error(f"Execution failed: {e}")
             raise RuntimeError(f"Node execution failed: {e}") from e

# --- ComfyUI Registration Info ---
# NODE_CLASS_MAPPINGS = { "FluxSamplerParameters": FluxSamplerParameters }
# NODE_DISPLAY_NAME_MAPPINGS = { "FluxSamplerParameters": "Flux Generate, Sample & Decode" }
