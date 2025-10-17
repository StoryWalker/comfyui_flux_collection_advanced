# /application/infrastructure/dependency_injection/container.py
__version__ = "1.4.0"
# 1.4.0 - Added providers for image saving.
# 1.3.0 - Added providers for VAE decoding.
# 1.2.0 - Added providers for sampling.
# 1.1.0 - Added providers for prompt encoding.
# 1.0.0 - Initial DI container setup.

from dependency_injector import containers, providers

# Import repositories (Infrastructure Layer)
from infrastructure.repositories.comfy_model_repository import ComfyModelRepository
from infrastructure.repositories.comfy_prompt_encoder_repository import ComfyPromptEncoderRepository
from infrastructure.repositories.comfy_sampler_repository import ComfySamplerRepository
from infrastructure.repositories.comfy_vae_repository import ComfyVaeRepository
from infrastructure.repositories.comfy_image_repository import ComfyImageRepository

# Import use cases (Application/Domain Layer)
from domain.use_cases import (
    LoadModelsUseCase,
    EncodeTextWithStylesUseCase,
    GenerateSampleAndDecodeUseCase,
    SaveImagesUseCase
)

class Container(containers.DeclarativeContainer):
    """
    Dependency Injection container for wiring the application components together.
    It maps abstract ports to concrete infrastructure implementations.
    """
    # --- Repositories (Concrete Implementations) ---
    model_repository = providers.Factory(ComfyModelRepository)
    prompt_encoder_repository = providers.Factory(ComfyPromptEncoderRepository)
    sampler_repository = providers.Factory(ComfySamplerRepository)
    vae_repository = providers.Factory(ComfyVaeRepository)
    image_repository = providers.Factory(ComfyImageRepository)

    # --- Use Cases (Application Logic) ---
    load_models_use_case = providers.Factory(
        LoadModelsUseCase,
        model_repository=model_repository
    )
    encode_text_use_case = providers.Factory(
        EncodeTextWithStylesUseCase,
        prompt_repository=prompt_encoder_repository
    )
    generate_sample_and_decode_use_case = providers.Factory(
        GenerateSampleAndDecodeUseCase,
        sampler_repository=sampler_repository,
        vae_repository=vae_repository
    )
    save_images_use_case = providers.Factory(
        SaveImagesUseCase,
        image_repository=image_repository
    )