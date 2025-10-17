# /application/infrastructure/comfyui_adapters/flux_models_loader_adapter.py
__version__ = "1.0.0"
# 1.0.0 - Refactored version of FluxModelsLoader.

import logging
from typing import Any, Dict, Tuple

# This is our new DI container
from infrastructure.dependency_injection.container import Container

# These are our pure domain data classes
from domain.entities import UnetConfig, ClipConfig, VaeConfig

# Configure logging
logger = logging.getLogger(__name__)

class FluxModelsLoaderRefactored:
    """
    Refactored version of the FluxModelsLoader node.
    This class acts as an adapter between the ComfyUI frontend and the
    application's core logic (use cases). Its responsibility is to translate
    UI inputs into domain objects and call the appropriate use case.
    """
    
    # We instantiate the container once
    container = Container()
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """
        Defines the input types for the node. It now uses the repository
        to fetch the model lists, keeping this class clean of filesystem logic.
        """
        logger.info("Setting up INPUT_TYPES for FluxModelsLoaderRefactored...")
        try:
            model_repo = cls.container.model_repository()
            unets, clips, vaes = model_repo.get_available_models()
        except Exception as e:
            logger.exception(f"Could not fetch model lists for node UI: {e}")
            unets, clips, vaes = [], [], []

        return {
            "required": {
                "unet_name": (unets, {"tooltip": "Name of the UNET/Diffusion model checkpoint file."}),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"tooltip": "Data type for UNET weights."}),
                "clip_name1": (clips, {"tooltip": "Name of the primary CLIP/Text Encoder model file."}),
                "clip_name2": (clips, {"tooltip": "Name of the secondary CLIP/Text Encoder model file."}),
                "type": (["flux", "sd3", "sdxl", "hunyuan_video"], {"tooltip": "Model architecture type for CLIP configuration."}),
                "vae_name": (vaes, {"tooltip": "Name of the VAE file or detected TAESD variant."}),
            },
            "optional": {
                "device": (["default", "cpu"], {"tooltip": "Force CLIP loading onto CPU (advanced)."}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE",)
    FUNCTION = "execute_refactored"
    CATEGORY = "flux_collection_advanced/refactored" # A new category to find it easily

    def execute_refactored(
        self, unet_name: str, weight_dtype: str, clip_name1: str, 
        clip_name2: str, type: str, vae_name: str, device: str = "default"
    ) -> Tuple[Any, Any, Any]:
        """
        Executes the refactored model loading logic.
        """
        node_name = self.__class__.__name__
        logger.info(f"Executing node: {node_name}")

        try:
            # 1. Create domain-specific configuration objects from UI inputs
            unet_config = UnetConfig(model_name=unet_name, weight_dtype=weight_dtype)
            clip_config = ClipConfig(
                clip_name1=clip_name1,
                clip_name2=clip_name2,
                architecture_type=type,
                device=device
            )
            vae_config = VaeConfig(vae_name=vae_name)

            # 2. Get the use case from the DI container
            load_models_uc = self.container.load_models_use_case()

            # 3. Execute the use case
            loaded_models = load_models_uc.execute(unet_config, clip_config, vae_config)
            
            logger.info(f"{node_name} execution completed successfully.")
            
            # 4. Return the results in the format ComfyUI expects
            return (loaded_models.unet, loaded_models.clip, loaded_models.vae)

        except Exception as e:
            logger.exception(f"A critical error occurred in {node_name}: {e}")
            raise # Re-raise the exception to show the error in the ComfyUI