# /application/infrastructure/comfyui_adapters/flux_sampler_parameters_adapter.py
__version__ = "1.1.1"
# 1.1.1 - Reordered inputs for better visual flow (model, positive, vae).
# 1.1.0 - Integrated VAE decoding, changing output from LATENT to IMAGE.
# 1.0.0 - Refactored version of FluxSamplerParameters.

import logging
from typing import Any, Dict, Tuple

import comfy.samplers
from infrastructure.dependency_injection.container import Container
from domain.entities import SamplerConfig, Conditioning

logger = logging.getLogger(__name__)

class FluxSamplerParametersRefactored:
    """
    Adapter for the FluxSamplerParameters node. It now handles the full
    pipeline from empty latent to final image by including VAE decoding.
    """
    container = Container()

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Defines the input types for the node."""
        return {
            "required": {
                # --- ORDEN DE ENTRADAS CORREGIDO ---
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "vae": ("VAE",),
                # ------------------------------------
                "width": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 28, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute_refactored"
    CATEGORY = "flux_collection_advanced/refactored"

    def execute_refactored(
        self, model: Any, positive: Any, vae: Any, width: int,
        height: int, batch_size: int, seed: int, steps: int,
        cfg: float, sampler_name: str, scheduler: str, denoise: float
    ) -> Tuple[Any,]:
        """Executes the refactored generate, sample, and decode logic."""
        # Note: The order of arguments here MUST match the new order in INPUT_TYPES.
        node_name = self.__class__.__name__
        logger.info(f"Executing node: {node_name} (full pipeline)")

        try:
            positive_conditioning = Conditioning(data=positive)
            sampler_config = SamplerConfig(
                model=model, seed=seed, steps=steps, cfg=cfg,
                sampler_name=sampler_name, scheduler=scheduler,
                positive_conditioning=positive_conditioning, denoise=denoise
            )

            use_case = self.container.generate_sample_and_decode_use_case()

            final_image = use_case.execute(
                width, height, batch_size, sampler_config, vae
            )

            logger.info(f"{node_name} full pipeline execution successful.")

            return (final_image.data,)

        except Exception as e:
            logger.exception(f"A critical error occurred in {node_name}: {e}")
            raise