# Task-Source: T#27
# -*- coding: utf-8 -*-
import os
import random
import json
import logging
import numpy as np
import torch
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
from .infrastructure.error_adapter import hex_error_handler

logger = logging.getLogger(__name__)

try:
    from comfy import cli_args
except ImportError:
    class MockArgs:
        disable_metadata = False
        disable_save_metadata = False
    cli_args = MockArgs()
    logger.warning("Could not import comfy.cli_args for ApplyControlNetPreview. Metadata saving defaults based on mock.")

_RANDOM_CHARS = "abcdefghijklmnopqrstuvwxyz"


class FluxControlNetApplyPreviewHex:
    """
    [HEX] ControlNet Apply & Preview — aplica ControlNet (positive only) y muestra un preview 
    de la imagen guía (hint image) directamente en el nodo.
    """

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "execute"
    CATEGORY = "flux_collection_advanced/hex"
    DESCRIPTION = (
        "[HEX] ControlNet Apply & Preview — aplica ControlNet (positive only) y muestra un preview\nde la imagen guía (hint image) directamente en el nodo.\n\n"
        "This documentation was AI-generated. If you find any errors or have suggestions for "
        "improvement, please feel free to contribute! Edit on GitHub."
    )
    OUTPUT_NODE = False

    _CONTROL_KEY = 'control'
    _APPLY_TO_UNCOND_KEY = 'control_apply_to_uncond'
    _VAE_KEY = 'vae'
    _DEFAULT_PREVIEW_PREFIX = "FluxACNPreview"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "section_controlnet": ("STRING", {"default": "CONTROLNET APPLY & PREVIEW"}),
                "positive": ("CONDITIONING", ),
                "vae": ("VAE", ),
                "image": ("IMAGE", {"tooltip": "The hint image to control conditioning and to preview."}),
                "control_net": ("CONTROL_NET", ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    def _validate_apply_inputs(self, positive, control_net, image, strength, start_percent, end_percent, vae):
        if not isinstance(positive, list): raise TypeError("Input 'positive' must be a list.")
        if not hasattr(control_net, 'copy') or not hasattr(control_net, 'set_cond_hint'): raise TypeError("Input 'control_net' invalid.")
        if not isinstance(image, torch.Tensor): raise TypeError("Input 'image' must be a Tensor.")
        if image.ndim != 4: raise ValueError(f"Input 'image' tensor wrong dimensions: {image.ndim} (expected 4).")
        if image.shape[0] == 0: raise ValueError("Input 'image' tensor is empty.")
        return True

    def _prepare_control_hint(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4: raise ValueError(f"Hint image tensor wrong dimensions: {image.ndim} (expected 4).")
        try:
            control_hint = image.movedim(-1, 1) # BHWC -> BCHW
            return control_hint
        except Exception as e:
            raise RuntimeError(f"Error processing hint image dimensions: {e}") from e

    def _process_single_conditioning(self, conditioning_list: list, base_control_net, control_hint: torch.Tensor, strength: float, timing: tuple[float, float], vae, control_net_cache: dict, extra_concat: list = []) -> list:
        processed_conditioning = []
        for i, conditioning_item in enumerate(conditioning_list):
            if not isinstance(conditioning_item, (list, tuple)) or len(conditioning_item) != 2: raise TypeError(f"Conditioning item {i} has invalid structure.")
            if not isinstance(conditioning_item[1], dict): raise TypeError(f"Conditioning item {i} dict missing.")

            tensor_data = conditioning_item[0]
            conditioning_dict_original = conditioning_item[1]
            conditioning_dict_copy = conditioning_dict_original.copy()

            try:
                previous_controlnet_ref = conditioning_dict_copy.get(self._CONTROL_KEY, None)
                if previous_controlnet_ref in control_net_cache:
                    final_control_net = control_net_cache[previous_controlnet_ref]
                else:
                    final_control_net = base_control_net.copy()
                    final_control_net = final_control_net.set_cond_hint(control_hint, strength, timing, vae=vae, extra_concat=extra_concat)
                    final_control_net.set_previous_controlnet(previous_controlnet_ref)
                    control_net_cache[previous_controlnet_ref] = final_control_net

                conditioning_dict_copy[self._CONTROL_KEY] = final_control_net
                conditioning_dict_copy[self._APPLY_TO_UNCOND_KEY] = False
                processed_conditioning.append([tensor_data, conditioning_dict_copy])

            except AttributeError as e:
                 raise AttributeError(f"Error applying ControlNet (check methods): {e}") from e
            except Exception as e:
                raise RuntimeError(f"Failed to process conditioning item {i}: {e}") from e
        return processed_conditioning

    @hex_error_handler
    def execute(self, **kwargs):
        positive = kwargs["positive"]
        vae = kwargs["vae"]
        image = kwargs["image"]
        control_net = kwargs["control_net"]
        strength = kwargs["strength"]
        start_percent = kwargs["start_percent"]
        end_percent = kwargs["end_percent"]
        extra_concat = kwargs.get("extra_concat", [])

        # --- 1. Apply ControlNet Logic ---
        processed_positive = []
        try:
            self._validate_apply_inputs(positive, control_net, image, strength, start_percent, end_percent, vae)

            if strength == 0:
                processed_positive = positive
            else:
                control_hint = self._prepare_control_hint(image)
                control_net_cache = {}
                timing = (start_percent, end_percent)
                processed_positive = self._process_single_conditioning(
                    positive, control_net, control_hint, strength, timing, vae, control_net_cache, extra_concat
                )

        except Exception as e:
            logger.exception(f"[HEX] Error during ControlNet application phase: {e}")
            processed_positive = positive

        # --- 2. Generate Preview Logic ---
        preview_results = []
        try:
            temp_output_dir = folder_paths.get_temp_directory()
            prefix_append = "_" + ''.join(random.choice(_RANDOM_CHARS) for _ in range(5))
            preview_prefix = self._DEFAULT_PREVIEW_PREFIX + prefix_append
            compress_level = 1

            os.makedirs(temp_output_dir, exist_ok=True)

            img_h, img_w = image.shape[1:3]
            full_output_folder, filename, counter, subfolder, filename_prefix_resolved = \
                folder_paths.get_save_image_path(preview_prefix, temp_output_dir, img_h, img_w)

            for i, image_tensor_in in enumerate(image):
                try:
                    img_array = 255. * image_tensor_in.cpu().numpy()
                    img_pil = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

                    metadata = None
                    file = f"{filename}_{counter:05}_.png"
                    full_path = os.path.join(full_output_folder, file)

                    img_pil.save(full_path, pnginfo=metadata, compress_level=compress_level)

                    preview_results.append({
                        "filename": file,
                        "subfolder": subfolder,
                        "type": "temp"
                    })
                    counter += 1

                except Exception as e_inner:
                    logger.exception(f"[HEX] Error processing/saving preview image index {i}: {e_inner}")
                    continue

        except Exception as e_outer:
            logger.exception(f"[HEX] Error during preview generation: {e_outer}")
            preview_results = []

        return (processed_positive, {"ui": {"images": preview_results}})
