# -*- coding: utf-8 -*-
import logging
import os # Needed for folder_paths functions used in INPUT_TYPES
from typing import Any, Dict, List, Tuple, Optional, Type

# Necessary third-party imports
import torch

# ComfyUI imports
import folder_paths
import comfy.utils
import comfy.model_management
import nodes # For inheritance

# Configure logging early
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Spandrel is used for advanced model loading
# Conditional import to allow node registration even if spandrel isn't installed
try:
    from spandrel import ModelLoader
    # from spandrel.types import ImageModel # Type hint not strictly needed now
    SPANDREL_AVAILABLE = True
    logger.debug("Spandrel library loaded successfully.")
except ImportError:
    SPANDREL_AVAILABLE = False
    # Define dummy class to prevent NameError if spandrel is missing
    class ModelLoader: pass
    logger.warning("Spandrel library not found. Upscaling with models will not work.")


# --- Main Node Class ---
class FluxImageUpscaler(nodes.ComfyNodeABC):
    """
    A ComfyUI node for upscaling images using either standard interpolation
    methods or dedicated upscaling models loaded via Spandrel.
    Handles model loading, memory management, tiling, and scaling factors.
    """

    # --- Node Metadata ---
    FUNCTION = "upscale_image"
    CATEGORY = "flux_collection_advanced"
    DESCRIPTION = "Upscales images using high-quality models (like 4x-UltraSharp) via Spandrel. Ideal for 2K/4K preparation."
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls: Type['FluxImageUpscaler']) -> Dict[str, Any]:
        """ Defines the input types for the node. """
        model_list = ["None"] # Default if spandrel is missing or no models found
        if SPANDREL_AVAILABLE:
            try:
                # Fetch model list, ensuring "None" is always an option
                models_in_folder = folder_paths.get_filename_list("upscale_models")
                model_list = ["None"] + models_in_folder
            except Exception as e:
                 logger.exception("Could not retrieve upscale model list.")
                 model_list = ["None", "Error: Could not list models"]

        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Image to upscale."}),
                "model_name": (model_list, {"tooltip": "Select upscale model (e.g. 4x-UltraSharp). 'None' uses interpolation."}),
                "upscale_method": (cls.upscale_methods, {"tooltip": "Standard interpolation method (for final adjustments)."}),
                "scale_by": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 16.0, "step": 0.01, "tooltip": "Overall scaling factor (e.g. 2.0 to double resolution)."}),
            }
        }

    # --- Instance Initialization ---
    def __init__(self):
        """ Initializes the node, setting initial model cache state. """
        self.loaded_model: Optional[Any] = None # Cache for the loaded Spandrel model
        self.current_model_name: Optional[str] = None # Name of the cached model

    # --- Core Logic Methods ---

    def _load_model(self, model_name: str) -> Optional[Any]:
        """
        Loads the specified upscale model using Spandrel. Caches the loaded model.
        Handles unloading previous model if a different one is selected.

        Args:
            model_name: The name of the model file in the 'upscale_models' directory, or "None".

        Returns:
            The loaded Spandrel model object, or None if loading failed or model_name is "None".

        Raises:
            RuntimeError: If Spandrel is required but not installed, or for loading errors.
            FileNotFoundError: If the specified model file cannot be located.
        """
        if model_name == "None":
            # If "None" is selected, ensure any previously loaded model is unloaded
            if self.loaded_model is not None:
                logger.debug(f"Unloading previously loaded model: {self.current_model_name}")
                # Allow garbage collection
                self.loaded_model = None
                self.current_model_name = None
            return None

        # Return cached model if it's the same as requested
        if self.loaded_model is not None and self.current_model_name == model_name:
            logger.debug(f"Using cached upscale model: {model_name}")
            return self.loaded_model

        # Unload previous model before loading a new one
        if self.loaded_model is not None:
             logger.debug(f"Unloading previous model '{self.current_model_name}' to load '{model_name}'.")
             self.loaded_model = None
             self.current_model_name = None
             # Suggest cache clearing, although Spandrel models might not use standard comfy cache
             comfy.model_management.soft_empty_cache()

        # Check if Spandrel is available before proceeding
        if not SPANDREL_AVAILABLE:
             raise RuntimeError("Spandrel library is required for model upscaling but is not installed.")

        logger.info(f"Loading upscale model: {model_name}")
        try:
            model_path = folder_paths.get_full_path("upscale_models", model_name)
            if not model_path: # Check if path resolution worked
                raise FileNotFoundError(f"Upscale model file '{model_name}' not found in expected directories.")

            # Use safe loading if available in comfy.utils
            if hasattr(comfy.utils, 'load_torch_file_safe'):
                 sd = comfy.utils.load_torch_file_safe(model_path)
            else:
                 sd = torch.load(model_path, map_location="cpu") # Load to CPU initially

            # Handle state dict prefix if present (common issue)
            if sd and next(iter(sd)).startswith("module."): # Check if sd is not empty first
                logger.debug("Replacing 'module.' prefix in model state_dict.")
                if hasattr(comfy.utils, 'state_dict_prefix_replace'):
                    sd = comfy.utils.state_dict_prefix_replace(sd, {"module.":""})
                else: # Manual fallback
                    sd = {k.replace("module.", ""): v for k, v in sd.items()}

            # Load using Spandrel's ModelLoader
            model = ModelLoader().load_from_state_dict(sd)
            model.eval() # Set to evaluation mode

            # Basic check for scale attribute (common but not guaranteed)
            if not hasattr(model, 'scale'):
                 logger.warning(f"Loaded model '{model_name}' does not have a 'scale' attribute. Assuming scale factor is 1.")

            # Cache the loaded model
            self.loaded_model = model
            self.current_model_name = model_name
            logger.info(f"Successfully loaded upscale model '{model_name}' (Detected Scale: {getattr(model, 'scale', 'N/A')}).")
            return self.loaded_model

        except FileNotFoundError as e:
             logger.error(f"Failed to find model file: {e}")
             raise # Re-raise specific file not found error
        except Exception as e:
            logger.exception(f"Failed to load upscale model '{model_name}': {e}")
            # Ensure cache is cleared on any loading error
            self.loaded_model = None
            self.current_model_name = None
            raise RuntimeError(f"Failed to load model '{model_name}': {e}") from e

    def _perform_model_upscale(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Performs tiled upscaling using the currently loaded Spandrel model.

        Args:
            image_tensor: Input image tensor [B, C, H, W] on the target device.

        Returns:
            Upscaled image tensor [B, C, H, W] on the target device.

        Raises:
            ValueError: If the model is not loaded or lacks a 'scale' attribute.
            RuntimeError: If tiling or model execution fails (e.g., OOM).
        """
        if self.loaded_model is None:
            raise ValueError("Cannot perform model upscale: Model is not loaded.")

        model_scale = getattr(self.loaded_model, 'scale', 1) # Default to 1 if no scale attr
        if model_scale <= 1:
            logger.warning(f"Model scale is {model_scale}. Model upscaling step might not change dimensions.")
            # Return input tensor directly if model scale won't change size
            # This prevents unnecessary processing and potential errors in tiled_scale
            return image_tensor

        logger.info(f"Performing model upscale (x{model_scale}) using '{self.current_model_name}'...")

        # Tiling parameters
        tile = 512
        overlap = 32
        oom = True

        while oom:
            try:
                # Calculate steps for progress bar
                steps = image_tensor.shape[0] * comfy.utils.get_tiled_scale_steps(
                    image_tensor.shape[3], image_tensor.shape[2], # W, H for tiled_scale_steps
                    tile_x=tile, tile_y=tile, overlap=overlap
                )
                pbar = comfy.utils.ProgressBar(steps) if comfy.utils.PROGRESS_BAR_ENABLED else None
                logger.debug(f"Tiled upscale starting with tile size {tile}x{tile}, overlap {overlap}.")

                # Perform tiled scaling
                upscaled_image = comfy.utils.tiled_scale(
                    image_tensor,
                    lambda x: self.loaded_model(x), # Pass model's forward method
                    tile_x=tile,
                    tile_y=tile,
                    overlap=overlap,
                    upscale_amount=model_scale, # Use the model's scale factor
                    pbar=pbar,
                )
                oom = False # If successful, exit loop
                logger.debug("Tiled upscale finished.")
            except comfy.model_management.OOM_EXCEPTION as e:
                logger.warning(f"OOM error during tiled upscale with tile size {tile}. Reducing tile size.")
                comfy.model_management.soft_empty_cache() # Attempt to free memory
                tile //= 2 # Reduce tile size
                if tile < 128: # Set a minimum practical tile size
                    logger.error("OOM error persists even with minimum tile size (128). Aborting upscale.")
                    raise RuntimeError("Out of memory during tiled upscaling, even with minimum tile size.") from e
            except Exception as e:
                 logger.exception(f"Error during tiled upscaling execution: {e}")
                 raise RuntimeError(f"Tiled upscaling failed: {e}") from e

        return upscaled_image

    def _final_scale(self, image_tensor: torch.Tensor, target_w: int, target_h: int, method: str) -> torch.Tensor:
        """
        Applies final scaling using standard interpolation methods (e.g., bilinear, lanczos).

        Args:
            image_tensor: The image tensor [B, H, W, C] (output from model or original).
            target_w: Final target width.
            target_h: Final target height.
            method: Interpolation method name.

        Returns:
            The final scaled image tensor [B, H, W, C].
        """
        current_h, current_w = image_tensor.shape[1:3] # Get current H, W from BHWC
        if current_w == target_w and current_h == target_h:
             logger.debug("Image already at final target size. Skipping final scaling.")
             return image_tensor # No scaling needed

        logger.info(f"Applying final scale from {current_w}x{current_h} to {target_w}x{target_h} using method '{method}'...")
        # comfy.utils.common_upscale expects BCHW format
        samples_in = image_tensor.movedim(-1, 1)
        samples_out = comfy.utils.common_upscale(samples_in, target_w, target_h, method, "disabled") # crop = "disabled"
        # Convert back to BHWC format for ComfyUI
        image_out = samples_out.movedim(1, -1)
        logger.debug("Final scaling applied.")
        return image_out

    # --- Main Execution Function ---
    def upscale_image(self, model_name: str, image: torch.Tensor, upscale_method: str, scale_by: float) -> Tuple[torch.Tensor,]:
        """
        Main execution function: Loads model (if specified), upscales image
        (using model primarily, then interpolation if needed), and applies
        the final scaling factor.

        Args:
            model_name: Name of the upscale model file selected or "None".
            image: Input image tensor [B, H, W, C].
            upscale_method: Standard interpolation method for final scaling adjustment.
            scale_by: The overall desired scaling factor relative to the original image.

        Returns:
            Tuple containing the final upscaled image tensor [B, H, W, C].
        """
        node_name = self.__class__.__name__
        logger.info(f"Executing node: {node_name}")

        # --- Input Validation ---
        if image is None or image.nelement() == 0:
            logger.error("Input image is missing or empty.")
            raise ValueError("Input image tensor is required.")
        if scale_by <= 0:
             logger.warning(f"Scale factor is {scale_by} (<= 0). Returning original image.")
             return (image,) # Return original image if scale factor is non-positive

        original_h, original_w = image.shape[1:3]
        target_h = round(original_h * scale_by)
        target_w = round(original_w * scale_by)

        # Handle potential zero target dimensions due to rounding very small inputs/scales
        if target_h <= 0 or target_w <= 0:
             logger.error(f"Calculated target dimensions are invalid ({target_w}x{target_h}). Check input image size and scale factor.")
             raise ValueError(f"Invalid target dimensions ({target_w}x{target_h}) calculated.")

        logger.info(f"Input size: {original_w}x{original_h}, Target size: {target_w}x{target_h} (Factor: {scale_by})")
        upscale_model = None # Ensure variable exists outside try

        try:
            # --- Step 1: Load Upscale Model (if selected) ---
            # This method handles caching and returns None if model_name is "None" or loading fails
            upscale_model = self._load_model(model_name)

            # --- Step 2: Perform Upscaling ---
            # Decide device strategy - use Comfy's preferred device
            processing_device = comfy.model_management.get_torch_device()
            logger.debug(f"Using processing device: {processing_device}")
            # Move input image to processing device only once
            current_image_tensor = image.to(processing_device)

            # --- Step 2a: Model Upscaling (if model loaded and scale > 1) ---
            if upscale_model is not None:
                model_scale = getattr(upscale_model, 'scale', 1) # Default to 1x if no scale
                logger.info(f"Using model '{model_name}' (native scale x{model_scale}).")

                if model_scale > 1: # Only run model if it actually scales up
                    try:
                        # Attempt memory management before moving model to device
                        memory_required = comfy.model_management.module_size(upscale_model.model)
                        comfy.model_management.free_memory(memory_required * 1.1, processing_device)
                        logger.debug(f"Attempted to free memory before model execution.")
                    except Exception as mem_e:
                        logger.warning(f"Memory check/free failed (continuing anyway): {mem_e}")

                    upscale_model.to(processing_device) # Move model to device for processing
                    input_for_model = current_image_tensor.movedim(-1, 1) # BHWC -> BCHW
                    upscaled_intermediate = self._perform_model_upscale(input_for_model) # Handles tiling & OOM
                    current_image_tensor = torch.clamp(upscaled_intermediate.movedim(1, -1), min=0, max=1.0) # BCHW -> BHWC & clamp
                    logger.info(f"Model upscale intermediate size: {current_image_tensor.shape[2]}x{current_image_tensor.shape[1]}")
                else:
                    logger.warning(f"Model '{model_name}' has scale <= 1. Skipping model upscale step, will use interpolation if needed.")
                # Move model back to CPU immediately after use or if skipped
                logger.debug(f"Moving model '{self.current_model_name}' back to CPU.")
                upscale_model.to("cpu")
                comfy.model_management.soft_empty_cache()
            else:
                # No model selected, will rely on final scaling step
                logger.info("No upscale model selected ('None'). Using standard interpolation if target size differs.")

            # --- Step 3: Final Scaling Adjustment (Interpolation) ---
            # Scale to the exact target size using the selected interpolation method
            # This is necessary if model scale != scale_by, or if no model was used.
            current_h, current_w = current_image_tensor.shape[1:3]
            if current_w != target_w or current_h != target_h:
                 upscaled_image = self._final_scale(current_image_tensor.to("cpu"), target_w, target_h, upscale_method) # Final scale on CPU
            else:
                 logger.info("Image already at target size. Skipping final scaling interpolation.")
                 upscaled_image = current_image_tensor.to("cpu") # Ensure output is on CPU

            final_h, final_w = upscaled_image.shape[1:3]
            logger.info(f"{node_name} execution completed. Final image size: {final_w}x{final_h}")

            # Final sanity check log
            if final_w == original_w and final_h == original_h and scale_by != 1.0:
                logger.warning(f"Final image size matches original size, but scale_by was {scale_by}. Review model scale and parameters.")

            return (upscaled_image,)

        except Exception as e:
            # General error handling, includes model cleanup attempt
            logger.exception(f"An error occurred during the upscaling process: {e}")
            # Ensure model is moved off GPU if an error occurs and it was loaded
            if self.loaded_model is not None:
                 try:
                      self.loaded_model.to("cpu")
                      logger.info("Moved model to CPU during error handling.")
                 except Exception as cleanup_e:
                      logger.error(f"Error moving model to CPU during cleanup: {cleanup_e}")
            raise RuntimeError(f"Upscaling process failed: {e}") from e


# --- ComfyUI Registration ---
# Example (in __init__.py):
#
# from .your_upscaler_node_file import FluxImageUpscaler
#
# NODE_CLASS_MAPPINGS = {
#    "FluxImageUpscaler": FluxImageUpscaler
# }
#
# NODE_DISPLAY_NAME_MAPPINGS = {
#    "FluxImageUpscaler": "Flux Image Upscaler (Spandrel)"
# }