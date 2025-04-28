# -*- coding: utf-8 -*-
import logging
from typing import Any, Dict, List, Tuple, Optional, Type

# Necessary third-party imports
import torch
# Spandrel is used for advanced model loading
try:
    from spandrel import ModelLoader
    # We don't strictly need ImageModel for type hints if we remove the isinstance check
    SPANDREL_AVAILABLE = True
except ImportError:
    SPANDREL_AVAILABLE = False
    class ModelLoader: pass # Dummy for code structure if not installed
    logger.warning("Spandrel library not found. Upscaling with models will not work.")


# ComfyUI imports
import folder_paths
import comfy.utils
import comfy.model_management
import nodes # For inheritance

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FluxImageUpscaler(nodes.ComfyNodeABC):
    """
    A ComfyUI node for upscaling images using either standard interpolation
    methods or dedicated upscaling models loaded via Spandrel.
    """

    # --- Node Metadata ---
    FUNCTION = "upscale_image"
    CATEGORY = "flux_collection_advanced"
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_NODE = False
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(cls: Type['FluxImageUpscaler']) -> Dict[str, Any]:
        """ Defines the input types for the node. """
        model_list = ["None"]
        if SPANDREL_AVAILABLE:
            try: model_list = ["None"] + folder_paths.get_filename_list("upscale_models")
            except Exception as e: logger.exception("Could not retrieve upscale model list."); model_list = ["None", "Error"]
        else: logger.warning("Spandrel not installed. Only standard methods available.")

        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (model_list, {"tooltip": "Upscale model ('None' for standard interpolation)."}),
                "upscale_method": (cls.upscale_methods, {"tooltip": "Interpolation method (used if model is 'None' or for final scaling)."}),
                "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01, "tooltip": "Factor to scale by."}),
            }
        }

    # --- Instance Initialization ---
    def __init__(self):
        self.loaded_model: Optional[Any] = None # Use Any, rely on Spandrel for type specifics
        self.current_model_name: Optional[str] = None

    # --- Core Logic Methods ---

    def _load_model(self, model_name: str) -> Optional[Any]:
        """ Loads the specified upscale model using Spandrel. Caches the loaded model. """
        if model_name == "None":
            if self.loaded_model is not None: logger.debug(f"Unloading previous model: {self.current_model_name}"); self.loaded_model = None; self.current_model_name = None
            return None
        if self.loaded_model is not None and self.current_model_name == model_name: return self.loaded_model
        if self.loaded_model is not None: logger.debug(f"Unloading '{self.current_model_name}' to load '{model_name}'."); self.loaded_model = None; self.current_model_name = None
        if not SPANDREL_AVAILABLE: raise RuntimeError("Spandrel library is required for model upscaling but is not installed.")

        logger.info(f"Loading upscale model: {model_name}")
        try:
            model_path = folder_paths.get_full_path("upscale_models", model_name)
            if not model_path: raise FileNotFoundError(f"Upscale model '{model_name}' not found.")

            if hasattr(comfy.utils, 'load_torch_file_safe'): sd = comfy.utils.load_torch_file_safe(model_path)
            else: sd = torch.load(model_path, map_location="cpu")

            # Handle potential state_dict prefix
            if next(iter(sd)).startswith("module."): # Simple check for prefix
                logger.debug("Replacing 'module.' prefix in state_dict.")
                if hasattr(comfy.utils, 'state_dict_prefix_replace'): sd = comfy.utils.state_dict_prefix_replace(sd, {"module.":""})
                else: sd = {k.replace("module.", ""): v for k, v in sd.items()}

            # Load using Spandrel - This should raise errors if the model is invalid
            model = ModelLoader().load_from_state_dict(sd).eval()

            # --- REMOVED isinstance check ---
            # if not isinstance(model, ImageModel) and not hasattr(model, 'scale'):
            #    logger.error(f"Loaded object for '{model_name}' is not a recognized Spandrel ImageModel.")
            #    raise ValueError(f"Model '{model_name}' is not a valid single-image upscale model.")
            # --- End of REMOVED check ---

            # Basic check based on expected attribute
            if not hasattr(model, 'scale'):
                 logger.warning(f"Loaded model '{model_name}' does not have a 'scale' attribute. Assuming scale=1.")
                 # You might choose to raise ValueError here if scale is mandatory

            self.loaded_model = model
            self.current_model_name = model_name
            logger.info(f"Successfully loaded upscale model '{model_name}' (Scale: {getattr(model, 'scale', 'N/A')}).")
            return self.loaded_model
        except FileNotFoundError as e: logger.error(f"Failed to find model file: {e}"); raise
        except Exception as e:
            logger.exception(f"Failed to load upscale model '{model_name}': {e}")
            self.loaded_model = None; self.current_model_name = None # Reset cache on error
            raise RuntimeError(f"Failed to load model '{model_name}': {e}") from e

    def _perform_model_upscale(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """ Performs tiled upscaling using the loaded Spandrel model. """
        # (Implementation unchanged)
        if self.loaded_model is None: raise ValueError("Model is not loaded.");
        model_scale = getattr(self.loaded_model, 'scale', 1) # Use getattr with default
        if model_scale == 1: logger.warning("Model scale is 1, model upscaling might not change dimensions.");
        logger.info(f"Performing model upscale (x{model_scale}) using {self.current_model_name}...")
        tile = 512; overlap = 32; oom = True;
        while oom:
            try:
                steps = image_tensor.shape[0] * comfy.utils.get_tiled_scale_steps(image_tensor.shape[3], image_tensor.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                pbar = comfy.utils.ProgressBar(steps) if comfy.utils.PROGRESS_BAR_ENABLED else None
                logger.debug(f"Tiled upscale: tile={tile}, overlap={overlap}.")
                upscaled_image = comfy.utils.tiled_scale(image_tensor, lambda x: self.loaded_model(x), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=model_scale, pbar=pbar,)
                oom = False; logger.debug("Tiled upscale finished.")
            except comfy.model_management.OOM_EXCEPTION as e:
                logger.warning(f"OOM during tiled upscale (tile={tile}). Reducing size."); comfy.model_management.soft_empty_cache(); tile //= 2;
                if tile < 128: logger.error("OOM persists even with tile size 128."); raise RuntimeError("Out of memory during tiled upscaling.") from e
            except Exception as e: logger.exception(f"Error during tiled upscaling: {e}"); raise RuntimeError(f"Tiled upscaling failed: {e}") from e
        return upscaled_image

    def _final_scale(self, image_tensor: torch.Tensor, target_w: int, target_h: int, method: str) -> torch.Tensor:
        """ Applies final scaling using standard interpolation methods. """
        # (Implementation unchanged)
        logger.info(f"Applying final scale to {target_w}x{target_h} using method '{method}'...")
        samples_in = image_tensor.movedim(-1, 1); samples_out = comfy.utils.common_upscale(samples_in, target_w, target_h, method, "disabled")
        image_out = samples_out.movedim(1, -1); logger.debug("Final scaling applied.")
        return image_out

    # --- Main Execution Function ---
    def upscale_image(self, model_name: str, image: torch.Tensor, upscale_method: str, scale_by: float) -> Tuple[torch.Tensor,]:
        """ Main function: Loads model (if specified), upscales, applies final scaling. """
        # (Implementation largely unchanged, ensures cleanup)
        node_name = self.__class__.__name__; logger.info(f"Executing node: {node_name}")
        if image is None or image.nelement() == 0: logger.error("Input image missing."); raise ValueError("Input image tensor required.")
        if scale_by <= 0: logger.warning(f"Scale factor <= 0 ({scale_by}), returning original."); return (image,)

        original_h, original_w = image.shape[1:3]; target_h = round(original_h * scale_by); target_w = round(original_w * scale_by)
        logger.info(f"Original: {original_w}x{original_h}, Target: {target_w}x{target_h} (Factor: {scale_by})")
        upscale_model = None # Define outside try block

        try:
            # Step 1: Load Model
            upscale_model = self._load_model(model_name) # Can return None or raise Error

            # Step 2: Upscale
            device = comfy.model_management.get_torch_device(); current_image_tensor = image.to(device)
            if upscale_model is not None:
                 # Model Upscaling
                 model_scale = getattr(upscale_model, 'scale', 1)
                 logger.info(f"Using model '{model_name}' (native x{model_scale}).")
                 try: # Manage memory/device for model processing
                      memory_required = comfy.model_management.module_size(upscale_model.model)
                      comfy.model_management.free_memory(memory_required * 1.1, device); logger.debug(f"Attempted free memory.")
                 except Exception as mem_e: logger.warning(f"Memory check/free failed: {mem_e}")
                 upscale_model.to(device)
                 input_for_model = current_image_tensor.movedim(-1, 1) # BHWC -> BCHW
                 upscaled_intermediate = self._perform_model_upscale(input_for_model)
                 logger.debug(f"Moving model '{self.current_model_name}' back to CPU.")
                 upscale_model.to("cpu"); comfy.model_management.soft_empty_cache() # Cleanup after use
                 current_image_tensor = torch.clamp(upscaled_intermediate.movedim(1, -1), min=0, max=1.0) # BCHW -> BHWC
                 logger.info(f"Model upscale intermediate size: {current_image_tensor.shape[2]}x{current_image_tensor.shape[1]}")
            else:
                 logger.info("No upscale model selected. Using standard interpolation for final scaling.")

            # Step 3: Final Scaling (if needed)
            current_h, current_w = current_image_tensor.shape[1:3]
            if current_w != target_w or current_h != target_h:
                upscaled_image = self._final_scale(current_image_tensor, target_w, target_h, upscale_method)
            else:
                logger.info("Image already at target size. Skipping final scaling.")
                upscaled_image = current_image_tensor

            upscaled_image = upscaled_image.to("cpu") # Ensure output is on CPU
            logger.info(f"{node_name} execution completed. Final size: {upscaled_image.shape[2]}x{upscaled_image.shape[1]}")
            return (upscaled_image,)

        except Exception as e:
            # Ensure model is moved off GPU if an error occurs during processing
            if self.loaded_model is not None:
                 try: self.loaded_model.to("cpu"); logger.info("Moved model to CPU during error handling.")
                 except Exception as cleanup_e: logger.error(f"Error moving model to CPU during cleanup: {cleanup_e}")
            logger.exception(f"An error occurred during upscaling process: {e}") # Log full traceback
            raise RuntimeError(f"Upscaling process failed: {e}") from e

# --- ComfyUI Registration ---
# Example:
# from .your_upscaler_node_file import FluxImageUpscaler
# NODE_CLASS_MAPPINGS = { "FluxImageUpscaler": FluxImageUpscaler }
# NODE_DISPLAY_NAME_MAPPINGS = { "FluxImageUpscaler": "Flux Image Upscaler (Spandrel)" }