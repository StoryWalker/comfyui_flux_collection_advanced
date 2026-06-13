# Task-Source: T#27
# -*- coding: utf-8 -*-
import logging
try:
    from infrastructure.doc_adapter import hex_node_doc
except ImportError:
    from .infrastructure.doc_adapter import hex_node_doc

import os
from typing import Any, Dict, Tuple, Optional, Type

import torch
import folder_paths
import comfy.utils
import comfy.model_management
import nodes
from .infrastructure.error_adapter import hex_error_handler

logger = logging.getLogger(__name__)

try:
    from spandrel import ModelLoader
    SPANDREL_AVAILABLE = True
except ImportError:
    SPANDREL_AVAILABLE = False
    class ModelLoader: pass
    logger.warning("Spandrel library not found. Upscaling with models will not work.")


@hex_node_doc
class FluxImageUpscalerHex:
    """[HEX] Image Upscaler — escala imágenes usando interpolación o modelos de súper resolución via Spandrel."""

    FUNCTION = "execute"
    CATEGORY = "flux_collection_advanced/hex"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = False

    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        model_list = ["None"]
        if SPANDREL_AVAILABLE:
            try:
                models_in_folder = folder_paths.get_filename_list("upscale_models")
                model_list = ["None"] + models_in_folder
            except Exception as e:
                 logger.exception("Could not retrieve upscale model list.")
                 model_list = ["None", "Error: Could not list models"]

        return {
            "required": {
                "section_upscaler": ("STRING", {"default": "UPSCALER"}),
                "image": ("IMAGE", {"tooltip": "Image to upscale."}),
                "model_name": (model_list, {"tooltip": "Select upscale model (e.g. 4x-UltraSharp). 'None' uses interpolation."}),
                "upscale_method": (cls.upscale_methods, {"tooltip": "Standard interpolation method (for final adjustments)."}),
                "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01, "tooltip": "Overall scaling factor (e.g. 2.0 to double resolution)."}),
            }
        }

    def __init__(self):
        self.loaded_model: Optional[Any] = None
        self.current_model_name: Optional[str] = None

    def _load_model(self, model_name: str) -> Optional[Any]:
        if model_name == "None":
            if self.loaded_model is not None:
                self.loaded_model = None
                self.current_model_name = None
            return None

        if self.loaded_model is not None and self.current_model_name == model_name:
            return self.loaded_model

        if self.loaded_model is not None:
             self.loaded_model = None
             self.current_model_name = None
             comfy.model_management.soft_empty_cache()

        if not SPANDREL_AVAILABLE:
             raise RuntimeError("Spandrel library is required for model upscaling but is not installed.")

        logger.info(f"[HEX] Loading upscale model: {model_name}")
        model_path = folder_paths.get_full_path("upscale_models", model_name)
        if not model_path:
            raise FileNotFoundError(f"Upscale model file '{model_name}' not found in expected directories.")

        if hasattr(comfy.utils, 'load_torch_file_safe'):
             sd = comfy.utils.load_torch_file_safe(model_path)
        else:
             sd = torch.load(model_path, map_location="cpu")

        if sd and next(iter(sd)).startswith("module."):
            if hasattr(comfy.utils, 'state_dict_prefix_replace'):
                sd = comfy.utils.state_dict_prefix_replace(sd, {"module.":""})
            else:
                sd = {k.replace("module.", ""): v for k, v in sd.items()}

        model = ModelLoader().load_from_state_dict(sd)
        model.eval()

        self.loaded_model = model
        self.current_model_name = model_name
        logger.info(f"[HEX] Successfully loaded upscale model '{model_name}' (Scale: {getattr(model, 'scale', 'N/A')}).")
        return self.loaded_model

    def _perform_model_upscale(self, image_tensor: torch.Tensor) -> torch.Tensor:
        if self.loaded_model is None:
            raise ValueError("Cannot perform model upscale: Model is not loaded.")

        model_scale = getattr(self.loaded_model, 'scale', 1)
        if model_scale <= 1:
            return image_tensor

        tile = 512
        overlap = 32
        oom = True

        while oom:
            try:
                steps = image_tensor.shape[0] * comfy.utils.get_tiled_scale_steps(
                    image_tensor.shape[3], image_tensor.shape[2],
                    tile_x=tile, tile_y=tile, overlap=overlap
                )
                pbar = comfy.utils.ProgressBar(steps) if comfy.utils.PROGRESS_BAR_ENABLED else None

                upscaled_image = comfy.utils.tiled_scale(
                    image_tensor,
                    lambda x: self.loaded_model(x),
                    tile_x=tile,
                    tile_y=tile,
                    overlap=overlap,
                    upscale_amount=model_scale,
                    pbar=pbar,
                )
                oom = False
            except comfy.model_management.OOM_EXCEPTION as e:
                logger.warning(f"[HEX] OOM during tiled upscale (tile size {tile}). Reducing tile size.")
                comfy.model_management.soft_empty_cache()
                tile //= 2
                if tile < 128:
                    raise RuntimeError("Out of memory during tiled upscaling, even with minimum tile size.") from e

        return upscaled_image

    def _final_scale(self, image_tensor: torch.Tensor, target_w: int, target_h: int, method: str) -> torch.Tensor:
        current_h, current_w = image_tensor.shape[1:3]
        if current_w == target_w and current_h == target_h:
             return image_tensor

        samples_in = image_tensor.movedim(-1, 1)
        samples_out = comfy.utils.common_upscale(samples_in, target_w, target_h, method, "disabled")
        return samples_out.movedim(1, -1)

    @hex_error_handler
    def execute(self, **kwargs) -> Tuple[torch.Tensor,]:
        model_name = kwargs["model_name"]
        image = kwargs["image"]
        upscale_method = kwargs["upscale_method"]
        scale_by = kwargs["scale_by"]

        if image is None or image.nelement() == 0:
            raise ValueError("Input image tensor is required.")
        if scale_by <= 0:
             return (image,)

        original_h, original_w = image.shape[1:3]
        target_h = round(original_h * scale_by)
        target_w = round(original_w * scale_by)

        if target_h <= 0 or target_w <= 0:
             raise ValueError(f"Invalid target dimensions ({target_w}x{target_h}) calculated.")

        upscale_model = self._load_model(model_name)
        processing_device = comfy.model_management.get_torch_device()
        current_image_tensor = image.to(processing_device)

        if upscale_model is not None:
            model_scale = getattr(upscale_model, 'scale', 1)
            if model_scale > 1:
                try:
                    memory_required = comfy.model_management.module_size(upscale_model.model)
                    comfy.model_management.free_memory(memory_required * 1.1, processing_device)
                except Exception as mem_e:
                    logger.warning(f"[HEX] Memory check/free failed (continuing anyway): {mem_e}")

                upscale_model.to(processing_device)
                input_for_model = current_image_tensor.movedim(-1, 1)
                upscaled_intermediate = self._perform_model_upscale(input_for_model)
                current_image_tensor = torch.clamp(upscaled_intermediate.movedim(1, -1), min=0.0, max=1.0)
            
            upscale_model.to("cpu")
            comfy.model_management.soft_empty_cache()

        current_h, current_w = current_image_tensor.shape[1:3]
        if current_w != target_w or current_h != target_h:
             upscaled_image = self._final_scale(current_image_tensor.to("cpu"), target_w, target_h, upscale_method)
        else:
             upscaled_image = current_image_tensor.to("cpu")

        return (upscaled_image,)
