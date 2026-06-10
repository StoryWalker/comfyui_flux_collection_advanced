# -*- coding: utf-8 -*-
"""
[NVFP4] Image Loader — Nodo experimental para carga de modelos NVFP4.
Compatible con checkpoints NVFP4 oficiales (FLUX.1/FLUX.2) en formato safetensors.
Requiere PyTorch CUDA 13.0+ y GPU Blackwell (RTX 50xx) para aceleración nativa.
"""
import logging

try:
    from domain.models import FluxModelConfig
    from application.nvfp4_image_generation_service import NVFP4ImageGenerationService
    from infrastructure.nvfp4_model_adapter import NVFP4ModelAdapter
    from infrastructure.error_adapter import ErrorLoggingAdapter, hex_error_handler
except ImportError:
    from .domain.models import FluxModelConfig
    from .application.nvfp4_image_generation_service import NVFP4ImageGenerationService
    from .infrastructure.nvfp4_model_adapter import NVFP4ModelAdapter
    from .infrastructure.error_adapter import ErrorLoggingAdapter, hex_error_handler

logger = logging.getLogger(__name__)


class NVFP4ImageLoader:
    """
    [NVFP4] Image Loader:
    Carga el stack completo UNET + CLIP + VAE con soporte para modelos NVFP4.
    ComfyUI detecta el formato NVFP4 automaticamente desde los metadatos del checkpoint.
    """

    @classmethod
    def INPUT_TYPES(cls):
        adapter = NVFP4ModelAdapter()
        unet_list = adapter.get_available_files(["diffusion_models", "unet"])
        clip_list = adapter.get_available_files(["clip", "text_encoders"])
        vae_list = adapter.get_available_files(["vae"])

        return {
            "required": {
                "section_model": ("STRING", {"default": "NVFP4 MODEL"}),
                "unet_name": (unet_list, {"tooltip": "UNET checkpoint (.safetensors con metadatos NVFP4, o .gguf)"}),
                "base_type": (["flux", "flux2"], {"default": "flux"}),

                "section_clip": ("STRING", {"default": "CLIP ENCODERS"}),
                "clip_name1": (clip_list, {"tooltip": "Encoder primario"}),
                "clip_name2": (["None"] + clip_list, {"tooltip": "Encoder secundario (T5-XXL). None para Flux 2."}),
                "clip_type": (["flux", "flux2"], {"default": "flux"}),

                "section_vae": ("STRING", {"default": "VAE"}),
                "vae_name": (vae_list, {"tooltip": "VAE estandar"}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "execute"
    CATEGORY = "flux_collection_advanced/nvfp4"

    @hex_error_handler
    def execute(self, **kwargs):
        config = FluxModelConfig(
            unet_name=kwargs.get("unet_name"),
            clip_name1=kwargs.get("clip_name1"),
            clip_name2=kwargs.get("clip_name2"),
            vae_name=kwargs.get("vae_name"),
            clip_type=kwargs.get("clip_type", "flux"),
            base_type=kwargs.get("base_type", "flux"),
            dequant_dtype="default",
            patch_dtype="default",
            patch_on_device=False,
        )

        service = NVFP4ImageGenerationService(
            model_adapter=NVFP4ModelAdapter(),
            error_adapter=ErrorLoggingAdapter(),
        )

        stack = service.load_stack(config)
        logger.info(f"[NVFP4] Stack cargado: {config.unet_name} | base={config.base_type}")
        return (stack["model"], stack["clip"], stack["vae"])
