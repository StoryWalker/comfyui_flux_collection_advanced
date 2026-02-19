import logging
from typing import Any, Dict, List, Tuple, Optional, Type
import comfy.controlnet
import folder_paths
import nodes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FluxControlNetLoader(nodes.ComfyNodeABC):
    """
    A ComfyUI node for loading ControlNet models specifically for Flux or other compatible architectures.
    This version takes the diffusion model as an input to ensure compatibility.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The diffusion model (UNET) that the ControlNet will be applied to."}),
                "control_net_name": (folder_paths.get_filename_list("controlnet"), {"tooltip": "The name of the ControlNet model file."})
            }
        }

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet"
    CATEGORY = "flux_collection_advanced"
    DESCRIPTION = "Loads a ControlNet model for use with Flux architectures."

    def load_controlnet(self, model, control_net_name):
        logger.info(f"Loading ControlNet model: {control_net_name}")
        try:
            controlnet_path = folder_paths.get_full_path_or_raise("controlnet", control_net_name)
            controlnet = comfy.controlnet.load_controlnet(controlnet_path, model)
            logger.info("ControlNet model loaded successfully.")
            return (controlnet,)
        except Exception as e:
            logger.exception(f"Failed to load ControlNet model '{control_net_name}': {e}")
            raise RuntimeError(f"Error loading ControlNet: {e}") from e

# Registration info for reference
# NODE_CLASS_MAPPINGS = { "FluxControlNetLoader": FluxControlNetLoader }
# NODE_DISPLAY_NAME_MAPPINGS = { "FluxControlNetLoader": "Flux ControlNet Loader" }
