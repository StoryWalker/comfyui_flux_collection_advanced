# /application/infrastructure/repositories/comfy_sampler_repository.py
__version__ = "1.0.1"
# 1.0.1 - Added centralized color logging.
# 1.0.0 - Concrete implementation of ISamplerRepository.

import logging
import torch
import comfy.model_management
import comfy.samplers
import comfy.sample
import comfy.utils
import latent_preview

from domain.ports import ISamplerRepository
from domain.entities import Latent, SamplerConfig
from infrastructure.utils.logging_colors import COLOR_BLUE, COLOR_RESET

logger = logging.getLogger(__name__)

# Determine default compute device for latents
# (This logic is copied from the original node)
try:
    if torch.cuda.is_available():
        DEFAULT_DEVICE = comfy.model_management.intermediate_device()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        DEFAULT_DEVICE = torch.device("mps")
    else:
        DEFAULT_DEVICE = torch.device("cpu")
except AttributeError:
    DEFAULT_DEVICE = comfy.model_management.text_encoder_device() if comfy.model_management.has_text_encoder_device() else torch.device("cpu")

# --- MENSAJE ACTUALIZADO CON COLOR ---
logger.info(f"{COLOR_BLUE}Using device for latent operations: {DEFAULT_DEVICE}{COLOR_RESET}")



class ComfySamplerRepository(ISamplerRepository):
    """
    Concrete repository that implements latent creation and sampling
    using ComfyUI's backend functions.
    """

    def create_empty_latent(self, width: int, height: int, batch_size: int) -> Latent:
        """See base class."""
        if width % 8 != 0 or height % 8 != 0:
            logger.warning(f"Width ({width}) or height ({height}) are not multiples of 8.")

        latent_channels = 16  # Specific to FLUX/SD3
        latent_height = height // 8
        latent_width = width // 8

        try:
            tensor = torch.zeros(
                [batch_size, latent_channels, latent_height, latent_width],
                device=DEFAULT_DEVICE
            )
            logger.info(f"Generated empty latent: {tensor.shape} on {tensor.device}")
            # ComfyUI expects latents in a dict format
            return Latent(samples={"samples": tensor})
        except Exception as e:
            logger.exception("Failed to generate empty latent tensor.")
            raise RuntimeError(f"Latent creation failed: {e}") from e

    def denoise_latent(self, initial_latent: Latent, config: SamplerConfig) -> Latent:
        """See base class."""
        try:
            latent_image = initial_latent.samples["samples"]
            model_device = config.model.load_device

            noise = comfy.sample.prepare_noise(latent_image, config.seed)
            noise = noise.to(model_device)

            preview_callback = latent_preview.prepare_callback(config.model, config.steps)
            disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

            logger.info(f"Starting sampling: {config.steps} steps, sampler={config.sampler_name}, scheduler={config.scheduler}, cfg={config.cfg}")

            # The original node passed an empty string for negative conditioning.
            # We access the positive conditioning data from the domain object.
            samples = comfy.sample.sample(
                model=config.model,
                noise=noise,
                steps=config.steps,
                cfg=config.cfg,
                sampler_name=config.sampler_name,
                scheduler=config.scheduler,
                positive=config.positive_conditioning.data,
                negative='',
                latent_image=latent_image.to(model_device),
                denoise=config.denoise,
                disable_noise=False,
                start_step=None,
                last_step=None,
                force_full_denoise=False,
                noise_mask=None,
                callback=preview_callback,
                disable_pbar=disable_pbar,
                seed=config.seed
            )

            samples_on_default_device = samples.to(DEFAULT_DEVICE)
            output_latent_dict = initial_latent.samples.copy()
            output_latent_dict["samples"] = samples_on_default_device

            logger.info("Sampling completed successfully.")
            return Latent(samples=output_latent_dict)

        except Exception as e:
            logger.exception(f"An error occurred during the sampling process: {e}")
            raise RuntimeError(f"Sampling process failed: {e}") from e