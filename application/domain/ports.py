# /application/domain/ports.py
__version__ = "1.4.1"
# 1.4.1 - Added method to IImageRepository for listing output subfolders.
# 1.4.0 - Added port for image saving.
# 1.3.0 - Added port for VAE decoding.
# 1.2.0 - Added port for latent generation and sampling.
# 1.1.0 - Added port for prompt encoding.
# 1.0.0 - Initial ports for model repositories.

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from .entities import (
    Conditioning, ClipConfig, LoadedModels, PromptConfig, Style,
    UnetConfig, VaeConfig, Latent, SamplerConfig,
    Image, ImageSaveConfig, ImageSaveResult
)

class IModelRepository(ABC):
    @abstractmethod
    def get_available_models(self) -> Tuple[List[str], List[str], List[str]]:
        pass

    @abstractmethod
    def load_all_models(
        self, unet_config: UnetConfig, clip_config: ClipConfig, vae_config: VaeConfig
    ) -> LoadedModels:
        pass

class IPromptEncoderRepository(ABC):
    @abstractmethod
    def get_available_styles(self) -> Dict[str, Style]:
        pass

    @abstractmethod
    def encode_prompt(self, config: PromptConfig, styles: Dict[str, Style]) -> Conditioning:
        pass

class ISamplerRepository(ABC):
    """
    Interface for a repository that handles latent creation and sampling.
    """

    @abstractmethod
    def create_empty_latent(self, width: int, height: int, batch_size: int) -> Latent:
        """
        Generates an empty latent tensor suitable for FLUX/SD3 models.

        Args:
            width: The target image width in pixels.
            height: The target image height in pixels.
            batch_size: The number of latent images to generate.

        Returns:
            A Latent object containing the empty tensor.
        """
        pass

    @abstractmethod
    def denoise_latent(self, initial_latent: Latent, config: SamplerConfig) -> Latent:
        """
        Performs the sampling (denoising) process on a given latent.

        Args:
            initial_latent: The starting latent tensor (can be empty or noisy).
            config: The configuration for the sampling process.

        Returns:
            A Latent object containing the final, denoised tensor.
        """
        pass

class IVaeRepository(ABC):
    """
    Interface for a repository that handles VAE operations like decoding.
    """
    @abstractmethod
    def decode_latent(self, vae_model: Any, latent: Latent) -> Image:
        """
        Decodes a latent tensor into a pixel-space image.

        Args:
            vae_model: The VAE model instance to use for decoding.
            latent: The Latent object to be decoded.

        Returns:
            An Image object containing the decoded tensor.
        """
        pass

class IImageRepository(ABC):
    """
    Interface for a repository that handles image persistence (saving).
    """
    @abstractmethod
    def get_available_output_subfolders(self) -> List[str]:
        """
        Retrieves a list of available subfolders within the main output directory.
        """
        pass

    @abstractmethod
    def save_image(self, image: Image, config: ImageSaveConfig) -> ImageSaveResult:
        """
        Saves a single image tensor to the filesystem.
        """
        pass