# /application/infrastructure/repositories/comfy_model_repository.py
__version__ = "1.0.1"
# 1.0.1 - Corrected CLIPType for SDXL from .SDXL to .STABLE_DIFFUSION.
# 1.0.0 - Concrete implementation of IModelRepository using ComfyUI's backend.

import logging
from typing import Any, Dict, List, Tuple

import torch
import comfy.sd
import comfy.utils
import folder_paths

from domain.ports import IModelRepository
from domain.entities import UnetConfig, ClipConfig, VaeConfig, LoadedModels

# Configure logging
logger = logging.getLogger(__name__)

class ComfyModelRepository(IModelRepository):
    """
    A concrete implementation of the IModelRepository that uses the ComfyUI
    backend functions for finding and loading models. This is an infrastructure adapter.
    """

    _TAESD_VARIANTS_PREFIXES = {
        "taesd": ["taesd_encoder.", "taesd_decoder."],
        "taesdxl": ["taesdxl_encoder.", "taesdxl_decoder."],
        "taesd3": ["taesd3_encoder.", "taesd3_decoder."],
        "taef1": ["taef1_encoder.", "taef1_decoder."],
    }
    _TAESD_SCALING = {
        "taesd": {"scale": 0.18215, "shift": 0.0},
        "taesdxl": {"scale": 0.13025, "shift": 0.0},
        "taesd3": {"scale": 1.5305, "shift": 0.0609},
        "taef1": {"scale": 0.3611, "shift": 0.1159},
    }

    def get_available_models(self) -> Tuple[List[str], List[str], List[str]]:
        """See base class."""
        try:
            unet_list = folder_paths.get_filename_list("diffusion_models")
            clip_list = folder_paths.get_filename_list("text_encoders")
            vae_list = self._get_available_vae_list()
            return (unet_list, clip_list, vae_list)
        except Exception as e:
            logger.exception("Failed to get model lists from ComfyUI.")
            return ([], [], [])

    def load_all_models(
        self,
        unet_config: UnetConfig,
        clip_config: ClipConfig,
        vae_config: VaeConfig
    ) -> LoadedModels:
        """See base class."""
        try:
            unet = self._load_unet(unet_config)
            clip = self._load_clip(clip_config)
            vae = self._load_vae(vae_config)

            return LoadedModels(unet=unet, clip=clip, vae=vae)
        except (FileNotFoundError, ValueError, KeyError, RuntimeError) as e:
            logger.error(f"Execution failed during model loading: {e}", exc_info=True)
            raise

    def _get_available_vae_list(self) -> List[str]:
        """Gets a list of available VAE names, including detected TAESD options."""
        vaes = folder_paths.get_filename_list("vae")
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        available_taesd = []
        for variant, prefixes in self._TAESD_VARIANTS_PREFIXES.items():
            has_encoder = any(v.startswith(prefixes[0]) for v in approx_vaes)
            has_decoder = any(v.startswith(prefixes[1]) for v in approx_vaes)
            if has_encoder and has_decoder:
                available_taesd.append(variant)
        return vaes + available_taesd

    def _load_unet(self, config: UnetConfig) -> Any:
        """Loads the UNET model."""
        logger.info(f"Loading UNET '{config.model_name}' with dtype '{config.weight_dtype}'...")
        model_options: Dict[str, Any] = {}
        if config.weight_dtype == "fp8_e4m3fn": model_options["dtype"] = torch.float8_e4m3fn
        elif config.weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif config.weight_dtype == "fp8_e5m2": model_options["dtype"] = torch.float8_e5m2
        
        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", config.model_name)
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        logger.info("UNET loaded successfully.")
        return model

    def _load_clip(self, config: ClipConfig) -> Any:
        """Loads the CLIP models."""
        logger.info(f"Loading CLIP models '{config.clip_name1}' and '{config.clip_name2}'...")
        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", config.clip_name1)
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", config.clip_name2)

        clip_type_map = {
            "flux": comfy.sd.CLIPType.FLUX,
            "sd3": comfy.sd.CLIPType.SD3,
            "sdxl": comfy.sd.CLIPType.STABLE_DIFFUSION, # <-- THE FIX IS HERE
            "hunyuan_video": comfy.sd.CLIPType.HUNYUAN_VIDEO,
        }
        clip_type = clip_type_map.get(config.architecture_type)
        if not clip_type:
            raise ValueError(f"Unsupported CLIP architecture: '{config.architecture_type}'")

        model_options: Dict[str, Any] = {}
        if config.device == "cpu":
            forced_device = torch.device("cpu")
            model_options["load_device"] = forced_device
            model_options["offload_device"] = forced_device

        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path1, clip_path2],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options=model_options
        )
        logger.info("CLIP models loaded successfully.")
        return clip

    def _load_vae(self, config: VaeConfig) -> Any:
        """Loads the VAE model."""
        logger.info(f"Loading VAE '{config.vae_name}'...")
        if config.vae_name in self._TAESD_VARIANTS_PREFIXES:
            state_dict = self._load_taesd_state_dict(config.vae_name)
        else:
            vae_path = folder_paths.get_full_path_or_raise("vae", config.vae_name)
            state_dict = comfy.utils.load_torch_file(vae_path)
        
        vae = comfy.sd.VAE(sd=state_dict)
        logger.info(f"VAE '{config.vae_name}' loaded successfully.")
        return vae

    def _load_taesd_state_dict(self, name: str) -> Dict[str, torch.Tensor]:
        """Loads a TAESD VAE state dictionary from components."""
        sd = {}
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        prefixes = self._TAESD_VARIANTS_PREFIXES[name]
        
        encoder_name = next(filter(lambda a: a.startswith(prefixes[0]), approx_vaes))
        decoder_name = next(filter(lambda a: a.startswith(prefixes[1]), approx_vaes))

        encoder_path = folder_paths.get_full_path_or_raise("vae_approx", encoder_name)
        enc_sd = comfy.utils.load_torch_file(encoder_path)
        for k, v in enc_sd.items(): sd[f"taesd_encoder.{k}"] = v

        decoder_path = folder_paths.get_full_path_or_raise("vae_approx", decoder_name)
        dec_sd = comfy.utils.load_torch_file(decoder_path)
        for k, v in dec_sd.items(): sd[f"taesd_decoder.{k}"] = v

        scale_info = self._TAESD_SCALING[name]
        sd["vae_scale"] = torch.tensor(scale_info["scale"])
        sd["vae_shift"] = torch.tensor(scale_info["shift"])
        return sd