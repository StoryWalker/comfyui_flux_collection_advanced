# -*- coding: utf-8 -*-
"""
[HEX] LTX Native Loader.
Orquesta los loaders nativos de ComfyUI para soportar Gemma en formato GGUF
(como archivo unico) en lugar de folder con archivos separados.

Este loader usa:
  - UnetLoaderGGUFAdvanced  (comfyui-gguf)
  - DualCLIPLoaderGGUF      (comfyui-gguf)
  - comfy.sd.VAE            (video VAE)
  - AudioVAE                (audio VAE)
  - LatentUpscaleModelLoader nativo
  - LoraLoaderModelOnly nativo (opcional)
"""
import logging
import os
import importlib

import comfy.sd
import comfy.utils
import folder_paths
import torch

try:
    from infrastructure.error_adapter import hex_error_handler
except ImportError:
    from .infrastructure.error_adapter import hex_error_handler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-import de nodos externos (pueden no estar disponibles en todos los entornos)
# ---------------------------------------------------------------------------
_GGUF_MODULE = None

def _get_gguf_module():
    global _GGUF_MODULE
    if _GGUF_MODULE is None:
        try:
            _GGUF_MODULE = importlib.import_module("custom_nodes.ComfyUI-GGUF.nodes")
        except Exception as e:
            logger.error(f"[LTX Native] No se pudo importar ComfyUI-GGUF: {e}")
            raise RuntimeError("ComfyUI-GGUF no esta instalado o no se pudo importar.")
    return _GGUF_MODULE


class LTXNativeLoaderHex:
    """
    [HEX] LTX Native Unified Loader.
    Carga el stack completo LTX 2.3 usando los loaders nativos de ComfyUI,
    permitiendo usar Gemma como archivo GGUF unico.
    """

    _CHECKPOINT_CATEGORIES = ["diffusion_models", "checkpoints", "unet"]
    _GGUF_CATEGORIES = ["diffusion_models", "checkpoints", "unet"]
    _CLIP_CATEGORIES = ["text_encoders", "clip"]
    _VAE_CATEGORIES = ["vae"]
    _UPSCALER_CATEGORIES = ["latent_upscale_models", "upscale_models"]
    _LORA_CATEGORIES = ["loras"]

    @classmethod
    def INPUT_TYPES(cls):
        gguf_mod = _get_gguf_module()

        # Listas de archivos disponibles
        unet_gguf = folder_paths.get_filename_list("unet_gguf") if hasattr(folder_paths, "get_filename_list") else []
        clip_gguf = folder_paths.get_filename_list("clip_gguf") if hasattr(folder_paths, "get_filename_list") else []
        text_enc = folder_paths.get_filename_list("text_encoders") if hasattr(folder_paths, "get_filename_list") else []
        vae_list = folder_paths.get_filename_list("vae") if hasattr(folder_paths, "get_filename_list") else []
        upscalers = folder_paths.get_filename_list("latent_upscale_models") if hasattr(folder_paths, "get_filename_list") else []
        loras = folder_paths.get_filename_list("loras") if hasattr(folder_paths, "get_filename_list") else []

        # Fallback: buscar en carpetas generales si las categorias custom no existen
        if not unet_gguf:
            unet_gguf = [f for f in folder_paths.get_filename_list("diffusion_models") if f.endswith(".gguf")]
        if not clip_gguf:
            clip_gguf = [f for f in text_enc if f.endswith(".gguf")]

        return {
            "required": {
                "section_models": ("STRING", {"default": "MODEL FILES"}),
                "unet_name": (sorted(list(set(unet_gguf))) if unet_gguf else ["None"],),
                "clip_name1": (sorted(list(set(clip_gguf))) if clip_gguf else ["None"],),
                "clip_name2": (sorted(list(set(text_enc))) if text_enc else ["None"],),
                "vae_video": (sorted(list(set(vae_list))) if vae_list else ["None"],),
                "vae_audio": (sorted(list(set(vae_list))) if vae_list else ["None"],),
                "upscaler": (sorted(list(set(upscalers))) if upscalers else ["None"],),
                "section_gguf": ("STRING", {"default": "GGUF OPTIONS"}),
                "dequant_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                "patch_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                "patch_on_device": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "lora": (sorted(list(set(loras))) if loras else ["None"],),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "VAE", "LATENT_UPSCALE_MODEL")
    RETURN_NAMES = ("model", "clip", "vae_video", "vae_audio", "upscale_model")
    FUNCTION = "execute"
    CATEGORY = "flux_collection_advanced/hex_ltx"

    @hex_error_handler
    def execute(self, **kwargs):
        unet_name = kwargs.get("unet_name", "None")
        clip_name1 = kwargs.get("clip_name1", "None")
        clip_name2 = kwargs.get("clip_name2", "None")
        vae_video_name = kwargs.get("vae_video", "None")
        vae_audio_name = kwargs.get("vae_audio", "None")
        upscaler_name = kwargs.get("upscaler", "None")
        lora_name = kwargs.get("lora", "None")
        lora_strength = kwargs.get("lora_strength", 1.0)

        dequant_dtype = kwargs.get("dequant_dtype", "default")
        patch_dtype = kwargs.get("patch_dtype", "default")
        patch_on_device = kwargs.get("patch_on_device", False)

        # -------------------------------------------------------------------
        # 1. Cargar UNET GGUF
        # -------------------------------------------------------------------
        gguf_mod = _get_gguf_module()
        unet_loader = gguf_mod.UnetLoaderGGUFAdvanced()
        model = unet_loader.load_unet(
            unet_name,
            dequant_dtype=dequant_dtype,
            patch_dtype=patch_dtype,
            patch_on_device=patch_on_device,
        )[0]
        logger.info(f"[LTX Native] UNET cargado: {unet_name}")

        # -------------------------------------------------------------------
        # 2. Cargar CLIP (Gemma GGUF + connectors)
        # -------------------------------------------------------------------
        clip_loader = gguf_mod.DualCLIPLoaderGGUF()
        clip = clip_loader.load_clip(clip_name1, clip_name2, type="ltxv")[0]
        logger.info(f"[LTX Native] CLIP cargado: {clip_name1} + {clip_name2}")

        # -------------------------------------------------------------------
        # 3. Cargar VAE de video (comfy nativo)
        # -------------------------------------------------------------------
        vae_video_path = self._resolve_path(vae_video_name, self._VAE_CATEGORIES)
        sd_video, metadata_video = comfy.utils.load_torch_file(vae_video_path, return_metadata=True)
        vae_video = comfy.sd.VAE(sd=sd_video, metadata=metadata_video)
        logger.info(f"[LTX Native] VAE Video cargado: {vae_video_name}")

        # -------------------------------------------------------------------
        # 4. Cargar VAE de audio (AudioVAE especializado)
        # -------------------------------------------------------------------
        try:
            from comfy.ldm.lightricks.vae.audio_vae import AudioVAE
        except ImportError:
            raise RuntimeError(
                "No se pudo importar AudioVAE. Asegurate de tener ComfyUI >= 0.3.68 con soporte LTXV."
            )

        vae_audio_path = self._resolve_path(vae_audio_name, self._VAE_CATEGORIES)
        sd_audio, metadata_audio = comfy.utils.load_torch_file(vae_audio_path, return_metadata=True)

        # AudioVAE requiere metadata con clave 'config'. Si falta, intentar leer directamente.
        if metadata_audio is None or "config" not in metadata_audio:
            try:
                import safetensors
                with safetensors.safe_open(vae_audio_path, framework="pt", device="cpu") as f:
                    metadata_audio = f.metadata()
            except Exception:
                pass

        if metadata_audio is None or "config" not in metadata_audio:
            raise RuntimeError(
                f"[LTX Native] El archivo de audio VAE no contiene metadata valida: {vae_audio_name}\n"
                f"Ruta: {vae_audio_path}\n\n"
                f"AudioVAE requiere metadata con clave 'config'. Posibles causas:\n"
                f"  1. El archivo esta corrupto o incompleto.\n"
                f"  2. No es el audio VAE oficial de LTX 2.3.\n"
                f"  3. El archivo deberia estar en formato .safetensors con metadata embebida.\n\n"
                f"Descarga el archivo oficial: ltx-2.3-22b-dev_audio_vae.safetensors"
            )

        vae_audio = AudioVAE(sd_audio, metadata_audio)
        logger.info(f"[LTX Native] VAE Audio cargado: {vae_audio_name}")

        # -------------------------------------------------------------------
        # 5. Cargar Upscaler latente (via nodo nativo)
        # -------------------------------------------------------------------
        from comfy_extras.nodes_hunyuan import LatentUpscaleModelLoader
        upscale_node = LatentUpscaleModelLoader()
        upscale_result = upscale_node.execute(upscaler_name)
        upscale_model = upscale_result.args[0]
        logger.info(f"[LTX Native] Upscaler cargado: {upscaler_name}")

        # -------------------------------------------------------------------
        # 6. Aplicar LoRA opcional al modelo
        # -------------------------------------------------------------------
        if lora_name and lora_name != "None":
            try:
                from nodes import LoraLoaderModelOnly
                lora_loader = LoraLoaderModelOnly()
                model = lora_loader.load_lora_model_only(model, lora_name, lora_strength)[0]
                logger.info(f"[LTX Native] LoRA aplicado: {lora_name} @ {lora_strength}")
            except Exception as e:
                logger.warning(f"[LTX Native] No se pudo aplicar LoRA {lora_name}: {e}")

        return (model, clip, vae_video, vae_audio, upscale_model)

    def _resolve_path(self, filename, folder_types):
        """Resuelve la ruta completa de un archivo en las carpetas de ComfyUI."""
        if not filename or filename == "None":
            return None
        if isinstance(folder_types, str):
            folder_types = [folder_types]
        for ft in folder_types:
            try:
                p = folder_paths.get_full_path(ft, filename)
                if p and os.path.exists(p):
                    return p
            except Exception:
                pass
        raise FileNotFoundError(f"[LTX Native] No se encontro el archivo: {filename} en {folder_types}")
