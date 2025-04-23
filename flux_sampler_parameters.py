import logging
from typing import Any, Dict, Tuple, Optional, Callable # Necessary types

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

# Determine default compute device for latents
try:
    if torch.cuda.is_available():
        DEFAULT_DEVICE = comfy.model_management.intermediate_device()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        DEFAULT_DEVICE = torch.device("mps")
    else:
        DEFAULT_DEVICE = torch.device("cpu")
except AttributeError:
     DEFAULT_DEVICE = comfy.model_management.text_encoder_device() if comfy.model_management.has_text_encoder_device() else torch.device("cpu")

logger.info(f"Using device for Flux/SD3 latent generation: {DEFAULT_DEVICE}")

class FluxSamplerParameters(nodes.ComfyNodeABC):
    """
    A ComfyUI node that generates an empty latent tensor suitable for
    Flux/SD3 type models and then samples (denoises) it using parameters
    and logic similar to KSampler, utilizing only positive conditioning.
    """

    # --- Node Metadata for ComfyUI ---
    FUNCTION = "generate_and_sample"
    CATEGORY = "flux_collection_advanced"
    DESCRIPTION = "Generates empty SD3/Flux latent and samples (positive only)."
    RETURN_TYPES = ("LATENT",)
    OUTPUT_TOOLTIPS = ("The final sampled (denoised) latent tensor.",)

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """ Define the required input types for the node. """
        model_type = "MODEL"
        conditioning_type = "CONDITIONING"

        return {
            "required": {
                "width": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16, "tooltip": "Width (pixels) of the latent image to generate."}),
                "height": ("INT", {"default": 1024, "min": 16, "max": nodes.MAX_RESOLUTION, "step": 16, "tooltip": "Height (pixels) of the latent image to generate."}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64, "tooltip": "Number of latent images to generate and sample in parallel."}),
                "model": (model_type, {"tooltip": "The model (e.g., Flux, SD3) to use for sampling."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True, "tooltip": "Random seed for noise generation."}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 10000, "tooltip": "Number of sampling (denoising) steps."}),
                "cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01, "tooltip": "Classifier-Free Guidance scale."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "Sampling algorithm to use."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "Noise scheduler."}),
                "positive": (conditioning_type, {"tooltip": "Positive conditioning (prompt embeddings)."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoising strength (1.0 = full sampling)."}),
            }
        }

    # --- Internal Methods (Core Logic) ---

    def _generate_empty_latent(self, width: int, height: int, batch_size: int) -> Dict[str, torch.Tensor]:
        """ Generates an empty latent tensor for Flux/SD3 (16 channels). """
        if width % 8 != 0 or height % 8 != 0:
            logger.warning(f"Width ({width}) or height ({height}) are not multiples of 8.")
        latent_channels = 16
        latent_height = height // 8
        latent_width = width // 8
        try:
            latent = torch.zeros([batch_size, latent_channels, latent_height, latent_width], device=DEFAULT_DEVICE)
            logger.info(f"Generated empty latent: {latent.shape} on {latent.device}")
            return {"samples": latent}
        except Exception as e:
            logger.exception("Failed to generate empty latent tensor.")
            raise RuntimeError(f"Latent creation failed: {e}") from e

    def _sample_latent(self, model: Any, seed: int, steps: int, cfg: float,
                       sampler_name: str, scheduler: str, positive: Any,
                       initial_latent: Dict[str, torch.Tensor], denoise: float
                       ) -> Tuple[Dict[str, torch.Tensor],]:
        """
        Executes the sampling logic (KSampler) on the initial latent, using only
        positive conditioning. Adds checks for inputs before sampling.
        """
        try:
            latent_image = initial_latent["samples"]
            model_device = model.load_device
            logger.debug(f"Model device for sampling: {model_device}")

            # Prepare noise
            logger.debug(f"Preparing noise with seed: {seed}")
            noise = comfy.sample.prepare_noise(latent_image, seed)
            noise = noise.to(model_device)

            # Prepare callback for previews
            logger.debug("Preparing latent preview callback...")
            preview_callback: Optional[Callable] = latent_preview.prepare_callback(model, steps)
            logger.debug(f"Preview callback type: {type(preview_callback)}") # Log callback type

            # Progress bar setting
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

            # --- Input Validation & Logging before comfy.sample.sample ---
            if positive is None:
                logger.error("Positive conditioning input is None. Cannot proceed with sampling.")
                raise ValueError("Positive conditioning cannot be None for sampling.")

            logger.debug(f"Type of positive conditioning: {type(positive)}")
            # You might add more detailed logging for conditioning structure if needed, e.g.:
            # if isinstance(positive, list):
            #    logger.debug(f"Length of positive conditioning list: {len(positive)}")
            #    if len(positive) > 0:
            #         logger.debug(f"Type of first element in positive conditioning: {type(positive[0])}")

            logger.debug(f"Latent image shape before sampling: {latent_image.shape}, Device: {latent_image.device}")
            logger.debug(f"Noise shape before sampling: {noise.shape}, Device: {noise.device}")
            # --- End of Input Validation ---

            # Execute main sampling, passing None for negative conditioning
            logger.info(f"Starting sampling (positive only): {steps} steps, sampler={sampler_name}, scheduler={scheduler}, cfg={cfg}, denoise={denoise}")
            samples = comfy.sample.sample(
                model=model, noise=noise, steps=steps, cfg=cfg,
                sampler_name=sampler_name, scheduler=scheduler,
                positive=positive, negative='',
                latent_image=latent_image.to(model_device), # Ensure latent is on model device
                denoise=denoise, disable_noise=False, start_step=None, last_step=None,
                force_full_denoise=False, noise_mask=None,
                callback=preview_callback, disable_pbar=disable_pbar, seed=seed
            )

            # Check if sampler returned None unexpectedly (highly unlikely but defensive)
            if samples is None:
                 logger.error("comfy.sample.sample returned None unexpectedly.")
                 raise RuntimeError("Sampling failed: Sampler returned None.")

            # Move result back to default device and package output
            samples = samples.to(DEFAULT_DEVICE)
            out_latent = initial_latent.copy()
            out_latent["samples"] = samples
            logger.info("Sampling completed.")
            return (out_latent,)

        except Exception as e:
            # Log the specific error before re-raising the generic one
            logger.exception(f"Specific error during sampling: {e}")
            raise RuntimeError(f"Sampling process failed: {e}") from e # Re-raise error

    # --- Main Execution Function for ComfyUI ---
    def generate_and_sample(self, width: int, height: int, batch_size: int,
                            model: Any, seed: int, steps: int, cfg: float,
                            sampler_name: str, scheduler: str,
                            positive: Any, denoise: float = 1.0
                            ) -> Tuple[Dict[str, torch.Tensor],]:
        """
        Orchestrates the generation of the empty latent and its subsequent sampling
        (using only positive conditioning).
        """
        node_name = self.__class__.__name__
        logger.info(f"Executing node: {node_name}")
        try:
            logger.info("Step 1: Generating empty latent (Flux/SD3)...")
            initial_latent_dict = self._generate_empty_latent(width, height, batch_size)
            logger.info("Step 2: Starting sampling process (positive only)...")
            final_latent_tuple = self._sample_latent(
                model=model, seed=seed, steps=steps, cfg=cfg,
                sampler_name=sampler_name, scheduler=scheduler,
                positive=positive,
                initial_latent=initial_latent_dict, denoise=denoise
            )
            logger.info(f"{node_name} execution completed successfully.")
            return final_latent_tuple
        except (ValueError, RuntimeError) as e:
             logger.error(f"Execution failed in {node_name}: {e}")
             raise
        except Exception as e:
             logger.exception(f"Unexpected critical error in {node_name}: {e}")
             raise RuntimeError(f"Unexpected critical error in {node_name}: {e}") from e

# --- ComfyUI Registration ---
# Example:
# from .your_node_file import FluxSamplerParameters
# NODE_CLASS_MAPPINGS = { "FluxSamplerParameters": FluxSamplerParameters }
# NODE_DISPLAY_NAME_MAPPINGS = { "FluxSamplerParameters": "Flux Generate & Sample (Pos Only)" }