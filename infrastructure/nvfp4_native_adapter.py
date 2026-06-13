# -*- coding: utf-8 -*-
import logging
import json
import torch
import safetensors.torch
import folder_paths
import comfy.sd
import comfy.utils
from typing import Any

logger = logging.getLogger(__name__)

class NVFP4NativeAdapter:
    """
    Adaptador de infraestructura para carga nativa de modelos NVFP4.
    Implementa los requisitos de node_requirements.md:
    1. Lee __metadata__["_quantization_metadata"].
    2. Fallback a .comfy_quant.
    3. Fallback heurístico.
    Inyecta los tensores en el state_dict al vuelo para comfy_kitchen.
    """

    @staticmethod
    def get_available_models():
        unet_list = folder_paths.get_filename_list("unet")
        diff_list = folder_paths.get_filename_list("diffusion_models")
        return sorted(list(set(unet_list + diff_list)))

    @staticmethod
    def detect_and_inject_quantization(state_dict: dict, metadata: dict) -> dict:
        quant_map = {}
        if metadata and "_quantization_metadata" in metadata:
            try:
                q_meta = json.loads(metadata["_quantization_metadata"])
                quant_map = q_meta.get("layers", {})
                logger.info("[NVFP4 Native] Header _quantization_metadata detectado.")
            except Exception as e:
                logger.warning(f"[NVFP4 Native] Error parseando _quantization_metadata: {e}")

        injected = 0
        keys = list(state_dict.keys())
        
        for key in keys:
            if not key.endswith(".weight"):
                continue
                
            prefix = key[:-7]
            layer_format = None
            
            # 1. Chequeo quant_map oficial
            if prefix in quant_map:
                layer_format = quant_map[prefix].get("format")
            
            # 2. Si ya tiene comfy_quant, saltamos
            comfy_quant_key = f"{prefix}.comfy_quant"
            if comfy_quant_key in state_dict:
                continue
                
            # 3. Heurística (uint8 + weight_scale + weight_scale_2 = nvfp4)
            if not layer_format:
                w = state_dict[key]
                has_ws = f"{prefix}.weight_scale" in state_dict
                has_ws2 = f"{prefix}.weight_scale_2" in state_dict
                if w.dtype == torch.uint8 and has_ws and has_ws2:
                    layer_format = "nvfp4"
                elif w.dtype in [torch.float8_e4m3fn, torch.float8_e5m2] and has_ws and not has_ws2:
                    layer_format = "float8_e4m3fn"

            # Inyectar .comfy_quant para el ecosistema ComfyUI / comfy_kitchen
            if layer_format:
                config_json = json.dumps({"format": layer_format}).encode("utf-8")
                state_dict[comfy_quant_key] = torch.tensor(list(config_json), dtype=torch.uint8)
                injected += 1

        if injected > 0:
            logger.info(f"[NVFP4 Native] Inyectados {injected} marcadores .comfy_quant al vuelo.")
            
        return state_dict

    @staticmethod
    def load_unet(unet_name: str) -> Any:
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name) or folder_paths.get_full_path("unet", unet_name)
        if not unet_path:
            raise FileNotFoundError(f"Modelo no encontrado: {unet_name}")

        logger.info(f"[NVFP4 Native] Iniciando carga de: {unet_name}")
        
        # Monkey-patch load_torch_file para interceptar el state_dict
        original_load_torch_file = comfy.utils.load_torch_file

        def custom_load_torch_file(*args, **kwargs):
            result = original_load_torch_file(*args, **kwargs)
            return_metadata = kwargs.get("return_metadata", False)
            if return_metadata:
                sd, comfy_metadata = result
            else:
                sd = result
                comfy_metadata = None

            metadata = {}
            if args[0].endswith(".safetensors"):
                with safetensors.safe_open(args[0], framework="pt") as f:
                    metadata = f.metadata() or {}
            
            sd = NVFP4NativeAdapter.detect_and_inject_quantization(sd, metadata)

            if return_metadata:
                return sd, comfy_metadata
            return sd

        try:
            comfy.utils.load_torch_file = custom_load_torch_file
            model = comfy.sd.load_diffusion_model(unet_path)
        finally:
            comfy.utils.load_torch_file = original_load_torch_file

        # Flux sampling shift
        if hasattr(model, "model") and hasattr(model.model, "model_sampling"):
            if not hasattr(model.model.model_sampling, "shift"):
                model.model.model_sampling.set_parameters(shift=1.15)
                
        return model
