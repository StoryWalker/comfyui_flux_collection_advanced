# Task-Source: T#12-GGUF
# -*- coding: utf-8 -*-
"""
Adaptador de infraestructura para la carga de modelos Flux GGUF y Safetensors.
Encapsula toda interaccion con ComfyUI, folder_paths y comfy.sd.
"""
import logging
import torch
import folder_paths
import comfy.sd
import comfy.utils
from typing import Any, List
from ..domain.models import FluxModelConfig

logger = logging.getLogger(__name__)

# Constantes de TAESD — escalas y prefijos por variante
_TAESD_PREFIXES = {
    "taesd":   ("taesd_encoder.",   "taesd_decoder."),
    "taesdxl": ("taesdxl_encoder.", "taesdxl_decoder."),
    "taesd3":  ("taesd3_encoder.",  "taesd3_decoder."),
    "taef1":   ("taef1_encoder.",   "taef1_decoder."),
}
_TAESD_SCALES = {
    "taesd":   (0.18215, 0.0),
    "taesdxl": (0.13025, 0.0),
    "taesd3":  (1.5305,  0.0609),
    "taef1":   (0.3611,  0.1159),
}


class FluxModelAdapter:
    """
    Adaptador de infraestructura para carga de modelos Flux.
    Soporta GGUF y Safetensors, single/dual CLIP, VAE estandar y TAESD.
    """

    @staticmethod
    def get_available_files(keys: List[str]) -> List[str]:
        """
        Agrega listas de archivos de multiples categorias de folder_paths.
        Retorna lista ordenada y deduplicada.
        """
        files = []
        for key in keys:
            try:
                files += folder_paths.get_filename_list(key)
            except Exception as e:
                logger.warning(f"[Flux] No se pudo listar archivos para '{key}': {e}")
        return sorted(set(files))

    @staticmethod
    def get_taesd_variant_names() -> List[str]:
        """ Detecta variantes TAESD disponibles en vae_approx. """
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        variants = []
        for name, (enc_prefix, _) in _TAESD_PREFIXES.items():
            if any(v.startswith(enc_prefix) for v in approx_vaes):
                variants.append(name)
        return variants

    @staticmethod
    def load_unet(config: FluxModelConfig) -> Any:
        """
        Carga UNET Flux desde GGUF o Safetensors.
        Usa UnetLoaderGGUFAdvanced para GGUF, comfy.sd para el resto.
        """
        logger.info(f"[Flux] Cargando UNET: {config.unet_name}")

        if config.is_gguf(config.unet_name):
            from nodes import NODE_CLASS_MAPPINGS
            gguf_node = NODE_CLASS_MAPPINGS.get("UnetLoaderGGUFAdvanced")
            if gguf_node is None:
                raise RuntimeError("Plugin ComfyUI-GGUF no encontrado. Instala comfyui-gguf.")
            return gguf_node().load_unet(
                config.unet_name, config.dequant_dtype,
                config.patch_dtype, config.patch_on_device
            )[0]

        unet_path = (
            folder_paths.get_full_path("diffusion_models", config.unet_name)
            or folder_paths.get_full_path("unet", config.unet_name)
        )
        if not unet_path:
            raise FileNotFoundError(f"UNET no encontrado: {config.unet_name}")
        return comfy.sd.load_diffusion_model(unet_path)

    @staticmethod
    def apply_flux_sampling(model: Any) -> None:
        """
        Aplica el shift de sampling de Flux (1.15) si el modelo lo soporta.
        No lanza excepcion si el modelo no tiene el atributo.
        """
        try:
            sampling = model.model.model_sampling
            if not hasattr(sampling, "shift"):
                logger.info("[Flux] Aplicando Flux Sampling Shift (1.15).")
                sampling.set_parameters(shift=1.15)
        except Exception as e:
            logger.warning(f"[Flux] No se pudo aplicar sampling shift: {e}")

    @staticmethod
    def load_clip(config: FluxModelConfig) -> Any:
        """
        Carga CLIP en modo single o dual, con soporte GGUF y Safetensors.
        Resuelve el tipo de CLIP via getattr para compatibilidad entre versiones de ComfyUI.
        """
        from nodes import NODE_CLASS_MAPPINGS
        c_type_str = config.clip_type_normalized
        c_type = getattr(comfy.sd.CLIPType, c_type_str.upper(), comfy.sd.CLIPType.FLUX)
        embeddings = folder_paths.get_folder_paths("embeddings")

        if config.use_single_clip:
            logger.info(f"[Flux] CLIP modo single: {config.clip_name1}")
            if config.is_gguf(config.clip_name1):
                node = NODE_CLASS_MAPPINGS.get("CLIPLoaderGGUF")
                if node is None:
                    raise RuntimeError("CLIPLoaderGGUF no encontrado. Instala comfyui-gguf.")
                return node().load_clip(config.clip_name1, c_type_str)[0]
            path = folder_paths.get_full_path_or_raise("clip", config.clip_name1)
            return comfy.sd.load_clip(ckpt_paths=[path], embedding_directory=embeddings, clip_type=c_type)

        # Modo dual
        logger.info(f"[Flux] CLIP modo dual: {config.clip_name1} + {config.clip_name2}")
        is_gguf = config.is_gguf(config.clip_name1) or config.is_gguf(config.clip_name2)
        if is_gguf:
            node = NODE_CLASS_MAPPINGS.get("DualCLIPLoaderGGUF")
            if node is None:
                raise RuntimeError("DualCLIPLoaderGGUF no encontrado. Instala comfyui-gguf.")
            return node().load_clip(config.clip_name1, config.clip_name2, c_type_str)[0]

        p1 = folder_paths.get_full_path_or_raise("text_encoders", config.clip_name1)
        p2 = folder_paths.get_full_path_or_raise("text_encoders", config.clip_name2)
        return comfy.sd.load_clip(ckpt_paths=[p1, p2], embedding_directory=embeddings, clip_type=c_type)

    @staticmethod
    def load_vae(vae_name: str) -> Any:
        """
        Carga VAE estandar o variante TAESD.
        Detecta automaticamente por nombre.
        """
        if vae_name in _TAESD_PREFIXES:
            return FluxModelAdapter._load_taesd(vae_name)

        logger.info(f"[Flux] Cargando VAE estandar: {vae_name}")
        vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
        return comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))

    @staticmethod
    def _load_taesd(name: str) -> Any:
        """ Ensambla un VAE TAESD desde sus archivos encoder/decoder separados. """
        logger.info(f"[Flux] Cargando TAESD: {name}")
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        enc_prefix, dec_prefix = _TAESD_PREFIXES[name]

        enc_name = next((v for v in approx_vaes if v.startswith(enc_prefix)), None)
        dec_name = next((v for v in approx_vaes if v.startswith(dec_prefix)), None)
        if not enc_name or not dec_name:
            raise FileNotFoundError(f"Archivos TAESD no encontrados para '{name}'.")

        enc_sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", enc_name))
        dec_sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", dec_name))

        sd = {}
        for k, v in enc_sd.items():
            sd[f"taesd_encoder.{k}"] = v
        for k, v in dec_sd.items():
            sd[f"taesd_decoder.{k}"] = v

        scale, shift = _TAESD_SCALES[name]
        sd["vae_scale"] = torch.tensor(scale)
        sd["vae_shift"] = torch.tensor(shift)
        return comfy.sd.VAE(sd=sd)
