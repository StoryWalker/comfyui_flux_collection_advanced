# -*- coding: utf-8 -*-
"""
Adaptador de infraestructura para modelos NVFP4.
Extiende FluxModelAdapter con detección automática de formatos NVFP4.
ComfyUI detecta NVFP4 nativamente via comfy_quant en el state_dict.
"""
import logging
import os
import torch
import folder_paths
import comfy.sd
import comfy.utils
from typing import Any, List

try:
    from domain.models import FluxModelConfig
except ImportError:
    from ..domain.models import FluxModelConfig

logger = logging.getLogger(__name__)


class NVFP4ModelAdapter:
    """
    Infrastructure adapter for NVFP4 model loading.
    NVFP4 is natively supported by ComfyUI when PyTorch is built with CUDA 13.0
    and the model contains .comfy_quant metadata in its state_dict.
    """

    @staticmethod
    def get_available_files(keys: List[str]) -> List[str]:
        """List available files from multiple folder_paths categories."""
        files = []
        for key in keys:
            try:
                files += folder_paths.get_filename_list(key)
            except Exception as e:
                logger.warning(f"[NVFP4] Could not list files for '{key}': {e}")
        return sorted(set(files))

    @staticmethod
    def load_unet(config: FluxModelConfig) -> Any:
        """
        Load UNET model. Supports NVFP4, GGUF, and standard SafeTensors.
        ComfyUI auto-detects NVFP4 quantization from state_dict metadata.
        """
        logger.info(f"[NVFP4] Loading UNET: {config.unet_name}")

        # NVFP4 models are typically distributed as .safetensors with quantization metadata
        if config.is_gguf(config.unet_name):
            from nodes import NODE_CLASS_MAPPINGS
            gguf_node = NODE_CLASS_MAPPINGS.get("UnetLoaderGGUFAdvanced")
            if gguf_node is None:
                raise RuntimeError("Plugin ComfyUI-GGUF not found. Install comfyui-gguf.")
            return gguf_node().load_unet(
                config.unet_name, config.dequant_dtype,
                config.patch_dtype, config.patch_on_device
            )[0]

        # Standard ComfyUI loader — auto-detects NVFP4 via detect_layer_quantization
        unet_path = (
            folder_paths.get_full_path("diffusion_models", config.unet_name)
            or folder_paths.get_full_path("unet", config.unet_name)
        )
        if not unet_path:
            raise FileNotFoundError(f"UNET not found: {config.unet_name}")

        model = comfy.sd.load_diffusion_model(unet_path)

        # Log if NVFP4 was detected
        if hasattr(model, 'model') and hasattr(model.model, 'model_config'):
            quant_config = getattr(model.model.model_config, 'quant_config', None)
            if quant_config:
                logger.info(f"[NVFP4] Quantization detected: {quant_config}")

        return model

    @staticmethod
    def load_clip(config: FluxModelConfig) -> Any:
        """Load CLIP encoder(s) with GGUF or standard format support."""
        from nodes import NODE_CLASS_MAPPINGS
        c_type_str = config.clip_type_normalized
        c_type = getattr(comfy.sd.CLIPType, c_type_str.upper(), comfy.sd.CLIPType.FLUX)
        embeddings = folder_paths.get_folder_paths("embeddings")

        if config.use_single_clip:
            logger.info(f"[NVFP4] CLIP single: {config.clip_name1}")
            if config.is_gguf(config.clip_name1):
                node = NODE_CLASS_MAPPINGS.get("CLIPLoaderGGUF")
                if node is None:
                    raise RuntimeError("CLIPLoaderGGUF not found. Install comfyui-gguf.")
                return node().load_clip(config.clip_name1, c_type_str)[0]
            path = folder_paths.get_full_path_or_raise("clip", config.clip_name1)
            return comfy.sd.load_clip(ckpt_paths=[path], embedding_directory=embeddings, clip_type=c_type)

        logger.info(f"[NVFP4] CLIP dual: {config.clip_name1} + {config.clip_name2}")
        is_gguf = config.is_gguf(config.clip_name1) or config.is_gguf(config.clip_name2)
        if is_gguf:
            node = NODE_CLASS_MAPPINGS.get("DualCLIPLoaderGGUF")
            if node is None:
                raise RuntimeError("DualCLIPLoaderGGUF not found. Install comfyui-gguf.")
            return node().load_clip(config.clip_name1, config.clip_name2, c_type_str)[0]

        p1 = folder_paths.get_full_path_or_raise("text_encoders", config.clip_name1)
        p2 = folder_paths.get_full_path_or_raise("text_encoders", config.clip_name2)
        return comfy.sd.load_clip(ckpt_paths=[p1, p2], embedding_directory=embeddings, clip_type=c_type)

    @staticmethod
    def load_vae(vae_name: str) -> Any:
        """Load VAE from standard safetensors."""
        logger.info(f"[NVFP4] Loading VAE: {vae_name}")
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        return comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))

    @staticmethod
    def apply_flux_sampling(model: Any) -> None:
        """Apply Flux sampling shift if applicable."""
        try:
            sampling = model.model.model_sampling
            if not hasattr(sampling, "shift"):
                logger.info("[NVFP4] Applying Flux Sampling Shift (1.15).")
                sampling.set_parameters(shift=1.15)
        except Exception as e:
            logger.warning(f"[NVFP4] Could not apply sampling shift: {e}")
