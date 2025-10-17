# /application/infrastructure/repositories/comfy_vae_repository.py
__version__ = "1.0.0"
# 1.0.0 - Concrete implementation of IVaeRepository.

import logging
from typing import Any
from domain.ports import IVaeRepository
from domain.entities import Latent, Image

logger = logging.getLogger(__name__)

class ComfyVaeRepository(IVaeRepository):
    """
    A concrete implementation of the IVaeRepository that uses the ComfyUI
    VAE object to perform decoding.
    """
    def decode_latent(self, vae_model: Any, latent: Latent) -> Image:
        """See base class."""
        logger.info("Decoding latent to image...")
        
        # This is the logic from the user's provided snippet
        samples = latent.samples.get("samples")
        if samples is None:
            raise ValueError("Latent dictionary is missing the 'samples' key.")

        images_tensor = vae_model.decode(samples)
        
        # Handle potential extra batch dimension
        if len(images_tensor.shape) == 5:
            logger.debug("Reshaping image tensor from 5D to 4D.")
            images_tensor = images_tensor.reshape(
                -1,
                images_tensor.shape[-3],
                images_tensor.shape[-2],
                images_tensor.shape[-1]
            )
        
        logger.info("Latent decoded successfully.")
        return Image(data=images_tensor)