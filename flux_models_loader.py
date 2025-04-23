import logging
from typing import Any, Dict, List, Tuple, Optional, Type, Callable # Added Callable just in case, kept Any

# Necessary third-party imports
import torch
import comfy.sd
import comfy.utils
import folder_paths
import nodes # For inheritance

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Determine default compute device for latents
try:
    if torch.cuda.is_available():
        DEFAULT_DEVICE = comfy.model_management.intermediate_device()
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        DEFAULT_DEVICE = torch.device("mps")
    else:
        DEFAULT_DEVICE = torch.device("cpu")
except AttributeError:
     DEFAULT_DEVICE = comfy.model_management.text_encoder_device() if comfy.model_management.has_text_encoder_device() else torch.device("cpu")

logger.info(f"Using device for Flux/SD3 latent generation: {DEFAULT_DEVICE}") # Log message is slightly inaccurate now, but device is still relevant

class FluxModelsLoader(nodes.ComfyNodeABC):
    """
    A ComfyUI node responsible for loading model components typically used
    with Flux or similar architectures (UNET, CLIP models, VAE).

    It handles different weight data types for the UNET, selects appropriate
    CLIP types, and supports standard VAEs as well as TAESD variants.
    """

    # --- Node Metadata for ComfyUI ---
    FUNCTION = "load_models"
    CATEGORY = "flux_collection_advanced"
    DESCRIPTION = "Loads UNET, CLIP, and VAE models for Flux/SD3 type architectures."
    RETURN_TYPES = ("MODEL", "CLIP", "VAE",)
    OUTPUT_NODE = False

    # --- Static Data for TAESD Handling ---
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

    @classmethod
    def INPUT_TYPES(cls: Type['FluxModelsLoader']) -> Dict[str, Any]:
        """ Defines the required and optional input types for the node. """
        try:
            unet_list = folder_paths.get_filename_list("diffusion_models")
            clip_list = folder_paths.get_filename_list("text_encoders")
            vae_list = cls._get_available_vae_list()
        except Exception as e:
            logger.exception("Failed to get model lists for INPUT_TYPES.")
            unet_list, clip_list, vae_list = [], [], []

        return {
            "required": {
                "unet_name": (unet_list, {"tooltip": "Name of the UNET/Diffusion model checkpoint file."}),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"], {"tooltip": "Data type for UNET weights."}),
                "clip_name1": (clip_list, {"tooltip": "Name of the primary CLIP/Text Encoder model file."}),
                "clip_name2": (clip_list, {"tooltip": "Name of the secondary CLIP/Text Encoder model file."}),
                "type": (["flux", "sd3", "sdxl", "hunyuan_video"], {"tooltip": "Model architecture type for CLIP configuration."}),
                "vae_name": (vae_list, {"tooltip": "Name of the VAE file or detected TAESD variant."}),
            },
            "optional": {
                "device": (["default", "cpu"], {"tooltip": "Force CLIP loading onto CPU (advanced)."}),
            }
        }

    # --- Static Helper Methods ---

    @classmethod
    def _get_available_vae_list(cls: Type['FluxModelsLoader']) -> List[str]:
        """ Gets a list of available VAE names, including detected TAESD options. """
        try:
            vaes = folder_paths.get_filename_list("vae")
            approx_vaes = folder_paths.get_filename_list("vae_approx")
            available_taesd = []
            for variant, prefixes in cls._TAESD_VARIANTS_PREFIXES.items():
                has_encoder = any(v.startswith(prefixes[0]) for v in approx_vaes)
                has_decoder = any(v.startswith(prefixes[1]) for v in approx_vaes)
                if has_encoder and has_decoder:
                    available_taesd.append(variant)
            return vaes + available_taesd
        except Exception as e:
             logger.exception("Failed to get VAE lists.")
             return []

    @classmethod
    def _load_taesd_state_dict(cls: Type['FluxModelsLoader'], name: str) -> Dict[str, torch.Tensor]:
        """ Loads a TAESD VAE state dictionary from components. """
        sd = {}
        try:
            approx_vaes = folder_paths.get_filename_list("vae_approx")
            prefixes = cls._TAESD_VARIANTS_PREFIXES[name]
            encoder_name = next(filter(lambda a: a.startswith(prefixes[0]), approx_vaes))
            decoder_name = next(filter(lambda a: a.startswith(prefixes[1]), approx_vaes))
            logger.info(f"Loading TAESD '{name}' using '{encoder_name}' and '{decoder_name}'.")

            encoder_path = folder_paths.get_full_path_or_raise("vae_approx", encoder_name)
            enc_sd = comfy.utils.load_torch_file(encoder_path)
            for k, v in enc_sd.items(): sd[f"taesd_encoder.{k}"] = v

            decoder_path = folder_paths.get_full_path_or_raise("vae_approx", decoder_name)
            dec_sd = comfy.utils.load_torch_file(decoder_path)
            for k, v in dec_sd.items(): sd[f"taesd_decoder.{k}"] = v

            scale_info = cls._TAESD_SCALING[name]
            sd["vae_scale"] = torch.tensor(scale_info["scale"])
            sd["vae_shift"] = torch.tensor(scale_info["shift"])
            return sd
        except StopIteration:
            err_msg = f"Missing encoder/decoder for TAESD variant '{name}' in 'vae_approx'."
            logger.error(err_msg)
            raise FileNotFoundError(err_msg)
        except KeyError:
            err_msg = f"Unrecognized TAESD variant name '{name}'."
            logger.error(err_msg)
            raise KeyError(err_msg)
        except FileNotFoundError as e:
             logger.error(f"TAESD file not found: {e}")
             raise
        except Exception as e:
            logger.exception(f"Failed to load TAESD state dict for '{name}'.")
            raise RuntimeError(f"Error loading TAESD '{name}': {e}") from e

    # --- Instance Methods (Loading Logic) ---

    # --- Type Hint Correction Here ---
    def _load_unet(self, unet_name: str, weight_dtype: str) -> Any: # Changed return type hint to Any
        """
        Loads the UNET/Diffusion model with specified weight precision.

        Args:
            unet_name: The name of the UNET model file.
            weight_dtype: The desired weight data type ("default", "fp8_e4m3fn", etc.).

        Returns:
            The loaded UNET model object (type varies by ComfyUI version).
        """
        logger.info(f"Loading UNET '{unet_name}' with dtype '{weight_dtype}'...")
        try:
            model_options: Dict[str, Any] = {}
            if weight_dtype == "fp8_e4m3fn": model_options["dtype"] = torch.float8_e4m3fn
            elif weight_dtype == "fp8_e4m3fn_fast":
                model_options["dtype"] = torch.float8_e4m3fn
                model_options["fp8_optimizations"] = True
                logger.info("Enabled fast FP8 optimizations for UNET.")
            elif weight_dtype == "fp8_e5m2": model_options["dtype"] = torch.float8_e5m2

            unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
            # The actual return type here is often ModelPatcher but we use Any for compatibility            
            model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
            logger.info("UNET loaded successfully.")
            return model
        except FileNotFoundError:
            logger.error(f"UNET file not found: {unet_name}")
            raise
        except Exception as e:
            logger.exception(f"Failed to load UNET '{unet_name}'.")
            raise RuntimeError(f"Error loading UNET: {e}") from e

    def _load_clip(self, clip_name1: str, clip_name2: str, model_arch_type: str, device_override: Optional[str]) -> Any: # Return type Any
        """
        Loads the CLIP (Text Encoder) models based on architecture type.

        Args:
            clip_name1: Name of the primary CLIP model file.
            clip_name2: Name of the secondary CLIP model file.
            model_arch_type: The model architecture type ("flux", "sd3", "sdxl", etc.).
            device_override: Optional device ("cpu") to force loading onto.

        Returns:
            The loaded CLIP object (type varies).
        """
        logger.info(f"Loading CLIP models '{clip_name1}' and '{clip_name2}' for type '{model_arch_type}'...")
        try:
            clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
            clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)

            if model_arch_type == "flux": clip_type = comfy.sd.CLIPType.FLUX
            elif model_arch_type == "sd3": clip_type = comfy.sd.CLIPType.SD3
            elif model_arch_type == "sdxl": clip_type = comfy.sd.CLIPType.SDXL
            elif model_arch_type == "hunyuan_video": clip_type = comfy.sd.CLIPType.HUNYUAN_VIDEO
            else: raise ValueError(f"Unsupported model architecture type for CLIP: '{model_arch_type}'")
            logger.debug(f"Determined CLIP type: {clip_type}")

            model_options: Dict[str, Any] = {}
            if device_override == "cpu":
                forced_device = torch.device("cpu")
                model_options["load_device"] = forced_device
                model_options["offload_device"] = forced_device
                logger.info("Forcing CLIP load/offload to CPU.")

            clip = comfy.sd.load_clip(
                ckpt_paths=[clip_path1, clip_path2],
                embedding_directory=folder_paths.get_folder_paths("embeddings"),
                clip_type=clip_type, model_options=model_options
            )
            logger.info("CLIP models loaded successfully.")
            return clip
        except FileNotFoundError:
            logger.error(f"CLIP file not found: {clip_name1} or {clip_name2}")
            raise
        except ValueError as e:
             logger.error(str(e))
             raise
        except Exception as e:
            logger.exception(f"Failed to load CLIP models.")
            raise RuntimeError(f"Error loading CLIP: {e}") from e

    def _load_vae(self, vae_name: str) -> Any: # Return type Any
        """
        Loads the VAE model, handling standard files and TAESD variants.

        Args:
            vae_name: Name of the VAE file or TAESD variant.

        Returns:
            The loaded VAE object (type varies).
        """
        logger.info(f"Loading VAE '{vae_name}'...")
        try:
            if vae_name in self._TAESD_VARIANTS_PREFIXES:
                state_dict = self._load_taesd_state_dict(vae_name)
            else:
                vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
                state_dict = comfy.utils.load_torch_file(vae_path)

            vae = comfy.sd.VAE(sd=state_dict)
            logger.info(f"VAE '{vae_name}' loaded successfully.")
            return vae
        except FileNotFoundError:
            logger.error(f"VAE file or TAESD component not found for '{vae_name}'")
            raise
        except KeyError as e:
             logger.error(f"VAE loading failed for '{vae_name}': {e}")
             raise RuntimeError(f"VAE loading failed: {e}") from e
        except Exception as e:
            logger.exception(f"Failed to load VAE '{vae_name}'.")
            raise RuntimeError(f"Error loading VAE: {e}") from e

    # --- Main Execution Function ---
    def load_models(self, unet_name: str, weight_dtype: str, clip_name1: str, clip_name2: str, type: str, vae_name: str, device: Optional[str] = "default") -> Tuple[Any, Any, Any]:
        """
        Loads the UNET, CLIP, and VAE models based on the provided inputs.

        Args:
            unet_name, weight_dtype, clip_name1, clip_name2, type, vae_name, device.

        Returns:
            Tuple containing the loaded (MODEL, CLIP, VAE).
        """
        node_name = self.__class__.__name__
        logger.info(f"Executing node: {node_name}")
        device_override = device if device != "default" else None

        try:
            if device_override not in [None, "cpu"]:
                 logger.warning(f"Invalid optional device '{device_override}', using default.")
                 device_override = None

            model = self._load_unet(unet_name, weight_dtype)
            clip = self._load_clip(clip_name1, clip_name2, type, device_override)
            vae = self._load_vae(vae_name)

            logger.info(f"{node_name} execution completed successfully.")
            return (model, clip, vae)

        except (FileNotFoundError, ValueError, KeyError, RuntimeError) as e:
             logger.error(f"Execution failed in {node_name}: {e}", exc_info=True)
             raise
        except Exception as e:
             logger.exception(f"An unexpected critical error occurred in {node_name}: {e}")
             raise RuntimeError(f"Unexpected critical error in {node_name}: {e}") from e

# --- ComfyUI Registration ---
# Example:
# from .your_loader_file import FluxModelsLoader
# NODE_CLASS_MAPPINGS = { "FluxModelsLoader": FluxModelsLoader }
# NODE_DISPLAY_NAME_MAPPINGS = { "FluxModelsLoader": "Flux Models Loader" }