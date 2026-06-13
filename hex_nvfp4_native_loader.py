# -*- coding: utf-8 -*-
import logging
try:
    from infrastructure.doc_adapter import hex_node_doc
except ImportError:
    from .infrastructure.doc_adapter import hex_node_doc


try:
    from domain.models import FluxModelConfig
    from application.nvfp4_native_service import NVFP4NativeService
    from infrastructure.nvfp4_native_adapter import NVFP4NativeAdapter
    from infrastructure.nvfp4_model_adapter import NVFP4ModelAdapter
    from infrastructure.error_adapter import hex_error_handler
except ImportError:
    from .domain.models import FluxModelConfig
    from .application.nvfp4_native_service import NVFP4NativeService
    from .infrastructure.nvfp4_native_adapter import NVFP4NativeAdapter
    from .infrastructure.nvfp4_model_adapter import NVFP4ModelAdapter
    from .infrastructure.error_adapter import hex_error_handler

logger = logging.getLogger(__name__)

@hex_node_doc
class NVFP4NativeLoaderHex:
    """
    Nodo ComfyUI para cargar checkpoints nativos NVFP4 de NVIDIA
    e interactuar directamente con comfy_kitchen en HW soportado.
    """
    @classmethod
    def INPUT_TYPES(cls):
        try:
            from infrastructure.model_filter_adapter import get_filtered_filenames
        except ImportError:
            from .infrastructure.model_filter_adapter import get_filtered_filenames
            
        unet_list = get_filtered_filenames("NVFP4NativeLoaderHex", "unet_name", "diffusion_models")
        clip_list1 = get_filtered_filenames("NVFP4NativeLoaderHex", "clip_name1", "clip")
        clip_list2 = get_filtered_filenames("NVFP4NativeLoaderHex", "clip_name2", "clip")
        vae_list = get_filtered_filenames("NVFP4NativeLoaderHex", "vae_name", "vae")

        return {
            "required": {
                "section_model": ("STRING", {"default": "NVFP4 NATIVE"}),
                "unet_name": (unet_list, {"tooltip": "NVFP4 UNET checkpoint (.safetensors oficial o convertido)"}),
                "base_type": (["flux", "flux2"], {"default": "flux"}),

                "section_clip": ("STRING", {"default": "CLIP ENCODERS"}),
                "clip_name1": (clip_list1, {"tooltip": "Primary CLIP"}),
                "clip_name2": (["None"] + clip_list2, {"tooltip": "Secondary CLIP (T5-XXL). None para Flux 2."}),
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

        service = NVFP4NativeService()
        stack = service.load_stack(config)
        
        logger.info(f"[NVFP4 Native] Carga finalizada.")
        return (stack["model"], stack["clip"], stack["vae"])
