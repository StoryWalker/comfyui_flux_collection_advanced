import torch # Assuming torch is used for IMAGE type
import logging # For logging potential issues or info

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FluxApplyControlNet:
    """
    Applies ControlNet conditioning to positive and negative conditioning tensors.

    This node takes existing conditioning data (positive and negative prompts/embeddings),
    a ControlNet model, and a hint image. It modifies the conditioning data
    by adding the ControlNet's influence, controlled by strength, start, and end percentages.
    It handles caching of ControlNet applications to potentially optimize workflows
    where the same base ControlNet is applied multiple times.
    """

    # --- Constants for dictionary keys ---
    _CONTROL_KEY = 'control'
    _APPLY_TO_UNCOND_KEY = 'control_apply_to_uncond'
    _VAE_KEY = 'vae' # Optional key from INPUT_TYPES

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input types and constraints for the node.

        Returns:
            dict: A dictionary describing required and optional inputs.
        """
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "control_net": ("CONTROL_NET", ),
                "image": ("IMAGE", ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                cls._VAE_KEY: ("VAE", ), # Use constant
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive", "negative")
    FUNCTION = "apply_controlnet"
    CATEGORY = "flux_collection_advanced" # Or perhaps "flux/conditioning/controlnet"? Adjust as needed.

    def _validate_inputs(self, positive, negative, control_net, image, strength, start_percent, end_percent, vae):
        """Validates the types and basic structure of input parameters."""
        if not isinstance(positive, list):
            raise TypeError("Input 'positive' must be a list (CONDITIONING).")
        if not isinstance(negative, list):
            raise TypeError("Input 'negative' must be a list (CONDITIONING).")
        # Basic check for control_net type - assumes it has necessary methods later
        if not hasattr(control_net, 'copy') or not hasattr(control_net, 'set_cond_hint'):
             raise TypeError("Input 'control_net' does not appear to be a valid CONTROL_NET object.")
        if not isinstance(image, torch.Tensor):
             raise TypeError("Input 'image' must be a torch.Tensor (IMAGE).")
        if not isinstance(strength, (float, int)):
            raise TypeError("Input 'strength' must be a float.")
        if not isinstance(start_percent, (float, int)):
            raise TypeError("Input 'start_percent' must be a float.")
        if not isinstance(end_percent, (float, int)):
            raise TypeError("Input 'end_percent' must be a float.")
        # VAE is optional, so check only if provided
        if vae is not None:
             # Add specific VAE checks if necessary, e.g., hasattr(vae, 'encode')
             pass
        if not (0.0 <= start_percent <= 1.0):
            raise ValueError(f"start_percent must be between 0.0 and 1.0, got {start_percent}")
        if not (0.0 <= end_percent <= 1.0):
            raise ValueError(f"end_percent must be between 0.0 and 1.0, got {end_percent}")
        if start_percent > end_percent:
            raise ValueError(f"start_percent ({start_percent}) cannot be greater than end_percent ({end_percent})")
        if not (0.0 <= strength <= 10.0): # Use max from INPUT_TYPES
             logger.warning(f"Strength ({strength}) is outside the typical range [0.0, 10.0].")
             # Allow execution but warn, or raise ValueError if strict adherence is required.
             # raise ValueError(f"Strength must be between 0.0 and 10.0, got {strength}")

    def _prepare_control_hint(self, image: torch.Tensor) -> torch.Tensor:
        """
        Prepares the control hint image by adjusting its dimensions.

        Args:
            image (torch.Tensor): The input hint image, typically (Batch, Height, Width, Channels).

        Returns:
            torch.Tensor: The processed hint image, typically (Batch, Channels, Height, Width).

        Raises:
            ValueError: If the input image tensor does not have the expected number of dimensions (e.g., 4 for BHWC).
        """
        if image.ndim != 4:
            # Assuming input is BHWC, target is BCHW. Adjust if format is different.
            raise ValueError(f"Expected input image tensor to have 4 dimensions (e.g., BHWC), but got {image.ndim}.")
        try:
            # Change dimension order from (..., H, W, C) to (..., C, H, W)
            control_hint = image.movedim(-1, 1)
            logger.info(f"Prepared control hint image with shape: {control_hint.shape}")
            return control_hint
        except Exception as e:
            logger.error(f"Failed to adjust dimensions for control hint image: {e}", exc_info=True)
            # Re-raise or raise a custom exception
            raise RuntimeError(f"Error processing hint image dimensions: {e}") from e

    def _process_single_conditioning(self,
                                     conditioning_list: list,
                                     base_control_net, # Renamed from control_net for clarity
                                     control_hint: torch.Tensor,
                                     strength: float,
                                     timing: tuple[float, float], # Combined start/end percent
                                     vae, # Pass VAE if needed by set_cond_hint
                                     control_net_cache: dict, # Cache for ControlNet instances
                                     extra_concat: list = []) -> list: # Include optional param if used
        """
        Processes a list of conditioning items (positive or negative) by applying the ControlNet.

        Args:
            conditioning_list (list): The list of conditioning items to process.
                                      Each item is expected to be [tensor, dict].
            base_control_net: The base ControlNet object to apply.
            control_hint (torch.Tensor): The prepared hint image.
            strength (float): The strength of the ControlNet effect.
            timing (tuple[float, float]): A tuple containing (start_percent, end_percent).
            vae: Optional VAE object passed to the ControlNet.
            control_net_cache (dict): A dictionary to store and retrieve previously configured ControlNet instances.
            extra_concat (list): Optional extra tensors for concatenation (passed to set_cond_hint).

        Returns:
            list: A new list with the processed conditioning items.

        Raises:
            TypeError: If a conditioning item does not have the expected structure [tensor, dict].
            AttributeError: If the base_control_net object lacks required methods.
            Exception: For unexpected errors during ControlNet application.
        """
        processed_conditioning = []
        for conditioning_item in conditioning_list:
            # --- Input Validation for conditioning item ---
            if not isinstance(conditioning_item, (list, tuple)) or len(conditioning_item) != 2:
                raise TypeError(f"Expected conditioning item to be a list/tuple of length 2, but got: {type(conditioning_item)}")
            if not isinstance(conditioning_item[1], dict):
                 raise TypeError(f"Expected second element of conditioning item to be a dict, but got: {type(conditioning_item[1])}")
            # --- End Validation ---

            tensor_data = conditioning_item[0]
            conditioning_dict_original = conditioning_item[1]

            # --- Deep Copy to avoid modifying original input ---
            # This is crucial for immutability and preventing side effects
            conditioning_dict_copy = conditioning_dict_original.copy()

            try:
                # Check cache based on previous control net reference (if any)
                previous_controlnet_ref = conditioning_dict_copy.get(self._CONTROL_KEY, None)

                if previous_controlnet_ref in control_net_cache:
                    # Reuse cached ControlNet instance
                    final_control_net = control_net_cache[previous_controlnet_ref]
                    logger.debug(f"Reusing cached ControlNet for previous_ref: {previous_controlnet_ref}")
                else:
                    # Create and configure a new ControlNet instance
                    logger.debug(f"Creating new ControlNet instance (previous_ref: {previous_controlnet_ref})")
                    # Copy the base ControlNet to avoid modifying it directly
                    final_control_net = base_control_net.copy()
                    # Apply settings
                    final_control_net = final_control_net.set_cond_hint(
                        control_hint,
                        strength,
                        timing, # Pass the tuple directly
                        vae=vae, # Pass optional VAE
                        extra_concat=extra_concat # Pass optional extra_concat
                    )
                    # Link to the previous one if it existed
                    final_control_net.set_previous_controlnet(previous_controlnet_ref)
                    # Store in cache
                    control_net_cache[previous_controlnet_ref] = final_control_net

                # Update the copied dictionary with the configured ControlNet
                conditioning_dict_copy[self._CONTROL_KEY] = final_control_net
                # Ensure ControlNet is not applied to unconditional conditioning by default here
                # Note: Some ControlNet implementations might handle this differently internally.
                conditioning_dict_copy[self._APPLY_TO_UNCOND_KEY] = False

                # Append the processed item (original tensor + modified dict) to the result list
                processed_conditioning.append([tensor_data, conditioning_dict_copy])

            except AttributeError as e:
                 logger.error(f"Object missing expected attribute/method during ControlNet processing: {e}", exc_info=True)
                 raise AttributeError(f"Error applying ControlNet: Ensure 'control_net' object has 'copy', 'set_cond_hint', and 'set_previous_controlnet' methods. Original error: {e}") from e
            except Exception as e:
                logger.error(f"An unexpected error occurred during processing of a conditioning item: {e}", exc_info=True)
                raise RuntimeError(f"Failed to process conditioning item: {e}") from e

        return processed_conditioning


    def apply_controlnet(self,
                         positive: list,
                         negative: list,
                         control_net,
                         image: torch.Tensor,
                         strength: float,
                         start_percent: float,
                         end_percent: float,
                         vae=None, # Keep optional VAE param
                         extra_concat: list = []): # Add missing but likely intended optional param
        """
        Applies the ControlNet to positive and negative conditioning lists.

        Args:
            positive (list): The positive conditioning list.
            negative (list): The negative conditioning list.
            control_net: The ControlNet model object.
            image (torch.Tensor): The hint image.
            strength (float): The strength of the ControlNet effect (0.0 to 10.0).
            start_percent (float): The timestep percentage at which to start applying the ControlNet (0.0 to 1.0).
            end_percent (float): The timestep percentage at which to stop applying the ControlNet (0.0 to 1.0).
            vae (optional): An optional VAE model, potentially used by the ControlNet. Defaults to None.
            extra_concat (list, optional): Optional extra tensors for concatenation, potentially used by ControlNet. Defaults to [].

        Returns:
            tuple[list, list]: A tuple containing the modified positive and negative conditioning lists.

        Raises:
            TypeError: If inputs have incorrect types.
            ValueError: If input values are outside valid ranges (e.g., percentages).
            AttributeError: If the control_net object is missing required methods.
            RuntimeError: For errors during image processing or ControlNet application.
        """
        logger.info(f"Executing FluxApplyControlNet with strength: {strength}, start: {start_percent}, end: {end_percent}")

        # --- Input Validation ---
        try:
             self._validate_inputs(positive, negative, control_net, image, strength, start_percent, end_percent, vae)
        except (TypeError, ValueError) as e:
            logger.error(f"Input validation failed: {e}", exc_info=True)
            # Re-raise the specific validation error
            raise e

        # --- Early Exit Condition ---
        if strength == 0:
            logger.info("Strength is 0, returning original conditioning.")
            return (positive, negative) # Return original inputs directly

        # --- Prepare Hint Image ---
        try:
            control_hint = self._prepare_control_hint(image)
        except (ValueError, RuntimeError) as e:
             # Error already logged in helper method
             # Re-raise the exception for ComfyUI/caller to handle
             raise e

        # --- Initialize Cache ---
        # Cache persists only for this execution run, mapping previous CN refs to newly configured ones
        control_net_cache = {}
        timing = (start_percent, end_percent)

        # --- Process Positive and Negative Conditioning ---
        try:
            processed_positive = self._process_single_conditioning(
                positive, control_net, control_hint, strength, timing, vae, control_net_cache, extra_concat
            )
            processed_negative = self._process_single_conditioning(
                negative, control_net, control_hint, strength, timing, vae, control_net_cache, extra_concat
            )
            logger.info("Successfully applied ControlNet to positive and negative conditioning.")
            return (processed_positive, processed_negative)

        except (TypeError, AttributeError, RuntimeError, Exception) as e:
             # Errors during processing are logged within _process_single_conditioning
             logger.error(f"ControlNet application failed: {e}", exc_info=True)
             # Re-raise the exception to halt execution and signal failure
             raise RuntimeError(f"Failed during ControlNet application: {e}") from e