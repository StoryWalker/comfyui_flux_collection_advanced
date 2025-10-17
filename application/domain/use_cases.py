# /application/domain/use_cases.py
__version__ = "1.4.0"
# 1.4.0 - Added use case for saving images.
# 1.3.1 - Fixed missing 'Any' type import.
# 1.3.0 - Added use case for the complete text-to-image pipeline (sample + decode).
# 1.2.0 - Added use case for generating and sampling latents.
# 1.1.0 - Added use case for encoding text prompts.
# 1.0.0 - Initial use case for loading models.

from typing import Any, List

from .entities import (
    UnetConfig, ClipConfig, VaeConfig, LoadedModels, PromptConfig,
    Conditioning, Latent, SamplerConfig,
    Image, ImageSaveConfig, ImageSaveResult
)
from .ports import (
    IModelRepository, IPromptEncoderRepository, ISamplerRepository, IVaeRepository, IImageRepository
)

class LoadModelsUseCase:
    """Orchestrates the loading of all necessary models for a workflow."""
    def __init__(self, model_repository: IModelRepository):
        self._model_repository = model_repository

    def execute(
        self, unet_config: UnetConfig, clip_config: ClipConfig, vae_config: VaeConfig
    ) -> LoadedModels:
        return self._model_repository.load_all_models(
            unet_config, clip_config, vae_config
        )

class EncodeTextWithStylesUseCase:
    """Orchestrates combining a text prompt with styles and encoding it."""
    def __init__(self, prompt_repository: IPromptEncoderRepository):
        self._prompt_repository = prompt_repository

    def execute(self, config: PromptConfig) -> Conditioning:
        available_styles = self._prompt_repository.get_available_styles()
        return self._prompt_repository.encode_prompt(config, available_styles)

class GenerateSampleAndDecodeUseCase:
    """
    Orchestrates the full pipeline:
    1. Creates an empty latent.
    2. Denoises the latent using a sampler.
    3. Decodes the final latent into an image.
    """
    def __init__(
        self,
        sampler_repository: ISamplerRepository,
        vae_repository: IVaeRepository
    ):
        self._sampler_repository = sampler_repository
        self._vae_repository = vae_repository

    def execute(
        self, width: int, height: int, batch_size: int,
        sampler_config: SamplerConfig, vae_model: Any
    ) -> Image:
        """Executes the full generate-sample-decode process."""
        empty_latent = self._sampler_repository.create_empty_latent(
            width, height, batch_size
        )

        final_latent = self._sampler_repository.denoise_latent(
            empty_latent, sampler_config
        )

        final_image = self._vae_repository.decode_latent(
            vae_model, final_latent
        )

        return final_image

class SaveImagesUseCase:
    """
    Orchestrates saving a batch of images to the filesystem.
    """
    def __init__(self, image_repository: IImageRepository):
        self._image_repository = image_repository

    def execute(self, images: List[Image], config: ImageSaveConfig) -> List[ImageSaveResult]:
        """
        Executes the save process for each image in the list.
        """
        results = []
        for image in images:
            result = self._image_repository.save_image(image, config)
            results.append(result)
        return results