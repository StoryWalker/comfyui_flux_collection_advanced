import torch
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FluxControlNetApply:
    """
    Applies ControlNet conditioning primarily to positive conditioning tensors,
    as typically used in Flux-based workflows.

    This node takes existing positive conditioning data, a ControlNet model,
    and a hint image. It modifies the positive conditioning by adding the
    ControlNet's influence, controlled by strength, start, and end percentages.
    It handles caching of ControlNet applications for potential optimization.
    Note: The 'negative' conditioning input/output has been removed based on
    typical Flux workflow patterns for this node type.
    """

    # --- Constants for dictionary keys ---
    _CONTROL_KEY = 'control'
    _APPLY_TO_UNCOND_KEY = 'control_apply_to_uncond'
    _VAE_KEY = 'vae'

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input types and constraints for the node.
        'negative' input removed.

        Returns:
            dict: A dictionary describing required and optional inputs.
        """
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                "image": ("IMAGE", ),
                "control_net": ("CONTROL_NET", ),                
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                cls._VAE_KEY: ("VAE", ),
            }
        }

    # Return only the modified positive conditioning
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("positive",)
    FUNCTION = "apply_controlnet"
    # --- CATEGORY updated as requested ---
    CATEGORY = "flux_collection_advanced"

    def _validate_inputs(self, positive, control_net, image, strength, start_percent, end_percent, vae):
        """
        Validates the types and basic structure of input parameters.
        'negative' validation removed.
        """
        if not isinstance(positive, list):
            raise TypeError("Input 'positive' must be a list (CONDITIONING).")
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
        if vae is not None:
             pass # Add specific checks if needed
        if not (0.0 <= start_percent <= 1.0):
            raise ValueError(f"start_percent must be between 0.0 and 1.0, got {start_percent}")
        if not (0.0 <= end_percent <= 1.0):
            raise ValueError(f"end_percent must be between 0.0 and 1.0, got {end_percent}")
        if start_percent > end_percent:
            raise ValueError(f"start_percent ({start_percent}) cannot be greater than end_percent ({end_percent})")
        if not (0.0 <= strength <= 10.0):
             logger.warning(f"Strength ({strength}) is outside the typical range [0.0, 10.0].")

    def _prepare_control_hint(self, image: torch.Tensor) -> torch.Tensor:
        """
        Prepares the control hint image by adjusting its dimensions.
        """
        if image.ndim != 4:
            raise ValueError(f"Expected input image tensor to have 4 dimensions (e.g., BHWC), but got {image.ndim}.")
        try:
            control_hint = image.movedim(-1, 1)
            logger.info(f"Prepared control hint image with shape: {control_hint.shape}")
            return control_hint
        except Exception as e:
            logger.error(f"Failed to adjust dimensions for control hint image: {e}", exc_info=True)
            raise RuntimeError(f"Error processing hint image dimensions: {e}") from e

    def _process_single_conditioning(self,
                                     conditioning_list: list,
                                     base_control_net,
                                     control_hint: torch.Tensor,
                                     strength: float,
                                     timing: tuple[float, float],
                                     vae,
                                     control_net_cache: dict,
                                     extra_concat: list = []) -> list:
        """
        Processes a list of conditioning items by applying the ControlNet.
        """
        processed_conditioning = []
        for conditioning_item in conditioning_list:
            if not isinstance(conditioning_item, (list, tuple)) or len(conditioning_item) != 2:
                raise TypeError(f"Expected conditioning item to be a list/tuple of length 2, but got: {type(conditioning_item)}")
            if not isinstance(conditioning_item[1], dict):
                 raise TypeError(f"Expected second element of conditioning item to be a dict, but got: {type(conditioning_item[1])}")

            tensor_data = conditioning_item[0]
            conditioning_dict_original = conditioning_item[1]
            conditioning_dict_copy = conditioning_dict_original.copy()

            try:
                previous_controlnet_ref = conditioning_dict_copy.get(self._CONTROL_KEY, None)

                if previous_controlnet_ref in control_net_cache:
                    final_control_net = control_net_cache[previous_controlnet_ref]
                    logger.debug(f"Reusing cached ControlNet for previous_ref: {previous_controlnet_ref}")
                else:
                    logger.debug(f"Creating new ControlNet instance (previous_ref: {previous_controlnet_ref})")
                    final_control_net = base_control_net.copy()
                    final_control_net = final_control_net.set_cond_hint(
                        control_hint, strength, timing, vae=vae, extra_concat=extra_concat
                    )
                    final_control_net.set_previous_controlnet(previous_controlnet_ref)
                    control_net_cache[previous_controlnet_ref] = final_control_net

                conditioning_dict_copy[self._CONTROL_KEY] = final_control_net
                conditioning_dict_copy[self._APPLY_TO_UNCOND_KEY] = False
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
                         control_net,
                         image: torch.Tensor,
                         strength: float,
                         start_percent: float,
                         end_percent: float,
                         vae=None,
                         extra_concat: list = []):
        """
        Applies the ControlNet to the positive conditioning list.

        Args:
            positive (list): The positive conditioning list.
            control_net: The ControlNet model object.
            image (torch.Tensor): The hint image.
            strength (float): The strength of the ControlNet effect (0.0 to 10.0).
            start_percent (float): The timestep percentage at which to start applying the ControlNet (0.0 to 1.0).
            end_percent (float): The timestep percentage at which to stop applying the ControlNet (0.0 to 1.0).
            vae (optional): An optional VAE model, potentially used by the ControlNet. Defaults to None.
            extra_concat (list, optional): Optional extra tensors for concatenation. Defaults to [].

        Returns:
            tuple[list]: A tuple containing only the modified positive conditioning list.

        Raises:
            TypeError: If inputs have incorrect types.
            ValueError: If input values are outside valid ranges.
            AttributeError: If the control_net object is missing required methods.
            RuntimeError: For errors during image processing or ControlNet application.
        """
        logger.info(f"Executing FluxControlNetApply (positive only) with strength: {strength}, start: {start_percent}, end: {end_percent}")

        # --- Input Validation (negative removed) ---
        try:
             self._validate_inputs(positive, control_net, image, strength, start_percent, end_percent, vae)
        except (TypeError, ValueError) as e:
            logger.error(f"Input validation failed: {e}", exc_info=True)
            raise e

        # --- Early Exit Condition ---
        if strength == 0:
            logger.info("Strength is 0, returning original positive conditioning.")
            return (positive,)

        # --- Prepare Hint Image ---
        try:
            control_hint = self._prepare_control_hint(image)
        except (ValueError, RuntimeError) as e:
             raise e

        # --- Initialize Cache ---
        control_net_cache = {}
        timing = (start_percent, end_percent)

        # --- Process Positive Conditioning Only ---
        try:
            processed_positive = self._process_single_conditioning(
                positive, control_net, control_hint, strength, timing, vae, control_net_cache, extra_concat
            )

            logger.info("Successfully applied ControlNet to positive conditioning.")
            return (processed_positive,)

        except (TypeError, AttributeError, RuntimeError, Exception) as e:
             logger.error(f"ControlNet application failed: {e}", exc_info=True)
             raise RuntimeError(f"Failed during ControlNet application: {e}") from e