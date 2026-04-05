# Task-Source: T#12-GGUF
# -*- coding: utf-8 -*-
import logging
from .domain.models import FluxModelConfig
from .application.image_loader_service import FluxLoaderService
from .infrastructure.image_model_adapter import FluxModelAdapter
from .infrastructure.error_adapter import ErrorLoggingAdapter, hex_error_handler

logger = logging.getLogger(__name__)


class FluxGGUFLoaderHex:
    """
    [HEX] Flux GGUF Advanced Loader.
    Carga el stack completo Flux (UNET + CLIP + VAE) con soporte GGUF y Safetensors.
    Compatible con Flux.1 y Flux.2. Soporta single CLIP para Flux 2.
    """

    @classmethod
    def INPUT_TYPES(cls):
        unet_list = FluxModelAdapter.get_available_files(["unet_gguf", "diffusion_models", "unet"])
        clip_list = FluxModelAdapter.get_available_files(["clip_gguf", "text_encoders", "clip"])
        vae_list  = FluxModelAdapter.get_available_files(["vae", "vae_approx"])
        vae_list += FluxModelAdapter.get_taesd_variant_names()

        return {
            "required": {
                # --- SECTION: UNET ---
                "section_unet":      ("STRING", {"default": "UNET MODEL"}),
                "unet_name":         (unet_list, {"tooltip": "UNET Flux (.gguf o .safetensors)"}),
                "base_type":         (["flux", "flux2", "wan2.1"], {"default": "flux"}),
                "dequant_dtype":     (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                "patch_dtype":       (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                "patch_on_device":   ("BOOLEAN", {"default": False}),

                # --- SECTION: CLIP ENCODERS ---
                "section_clip":      ("STRING", {"default": "CLIP ENCODERS"}),
                "clip_name1":        (clip_list, {"tooltip": "Encoder primario (CLIP-L o combinado)"}),
                "clip_name2":        (["None"] + clip_list, {"tooltip": "Encoder secundario (T5-XXL). 'None' para Flux 2."}),
                "clip_type":         (["flux", "flux2", "sd3", "sdxl"], {"default": "flux"}),

                # --- SECTION: VAE ---
                "section_vae":       ("STRING", {"default": "VAE"}),
                "vae_name":          (sorted(set(vae_list)), {"tooltip": "VAE o variante TAESD"}),
            }
        }

    RETURN_TYPES  = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES  = ("MODEL", "CLIP", "VAE")
    FUNCTION      = "execute"
    CATEGORY      = "flux_collection_advanced/hex"

    @hex_error_handler
    def execute(self, **kwargs):
        config = FluxModelConfig(
            unet_name      = kwargs.get("unet_name"),
            clip_name1     = kwargs.get("clip_name1"),
            clip_name2     = kwargs.get("clip_name2"),
            vae_name       = kwargs.get("vae_name"),
            clip_type      = kwargs.get("clip_type", "flux"),
            base_type      = kwargs.get("base_type", "flux"),
            dequant_dtype  = kwargs.get("dequant_dtype", "default"),
            patch_dtype    = kwargs.get("patch_dtype", "default"),
            patch_on_device= kwargs.get("patch_on_device", False),
        )

        service = FluxLoaderService(
            model_adapter=FluxModelAdapter(),
            error_adapter=ErrorLoggingAdapter(),
        )

        result = service.load_full_flux_stack(config)
        return (result["model"], result["clip"], result["vae"])
