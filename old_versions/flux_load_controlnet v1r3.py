# -*- coding: utf-8 -*-
import os
import logging
import importlib.util
from typing import Any, Dict, List, Tuple, Optional, Type

# Necessary third-party imports
import torch
import numpy as np
from PIL import Image, ImageOps, ImageSequence

# ComfyUI imports
import folder_paths
import nodes # For inheritance
import node_helpers # For safe PIL operations

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Main Node Class ---
class FluxLoadControlNetPreprocessor(nodes.ComfyNodeABC):
    """
    A ComfyUI node that loads an image (from file or upload), applies a
    selected ControlNet preprocessor, extracts an optional mask, and suggests
    a corresponding ControlNet model name pattern.
    It dynamically loads preprocessors from compatible custom node packs.
    """

    # --- Node Metadata ---
    FUNCTION = "execute"
    CATEGORY = "flux_collection_advanced" # Using the category from previous node
    RETURN_TYPES = ("IMAGE", "MASK", "STRING") # Added MASK output
    RETURN_NAMES = ("IMAGE", "MASK", "CNET_NAME_SUGGESTION") # Added MASK output name
    OUTPUT_NODE = False

    # --- Class Attributes for Preprocessor Management ---
    # (Same as previous version: _initialized, _available_preprocessors, etc.)
    _initialized: bool = False
    _available_preprocessors: List[str] = ["None"]
    _aio_preprocessor_instance: Optional[Any] = None
    _preprocessor_module_path: Optional[str] = None
    _PREPROCESSORS_CLASS_MAP: Dict[str, str] = { # Populated by _initialize_preprocessors
        "canny": "CannyEdgePreprocessor", "canny_pyra": "PyraCannyPreprocessor", "lineart": "LineArtPreprocessor",
        "lineart_anime": "AnimeLineArtPreprocessor", "lineart_manga": "Manga2Anime_LineArt_Preprocessor", "lineart_any": "AnyLineArtPreprocessor_aux",
        "scribble": "ScribblePreprocessor", "scribble_xdog": "Scribble_XDoG_Preprocessor", "scribble_pidi": "Scribble_PiDiNet_Preprocessor",
        "scribble_hed": "FakeScribblePreprocessor", "hed": "HEDPreprocessor", "pidi": "PiDiNetPreprocessor", "mlsd": "M-LSDPreprocessor",
        "pose": "DWPreprocessor", "openpose": "OpenposePreprocessor", "dwpose": "DWPreprocessor", "pose_dense": "DensePosePreprocessor",
        "pose_animal": "AnimalPosePreprocessor", "normalmap_bae": "BAE-NormalMapPreprocessor", "normalmap_dsine": "DSINE-NormalMapPreprocessor",
        "normalmap_midas": "MiDaS-NormalMapPreprocessor", "depth": "DepthAnythingV2Preprocessor", "depth_anything": "DepthAnythingPreprocessor",
        "depth_anything_v2": "DepthAnythingV2Preprocessor", "depth_anything_zoe": "Zoe_DepthAnythingPreprocessor", "depth_zoe": "Zoe-DepthMapPreprocessor",
        "depth_midas": "MiDaS-DepthMapPreprocessor", "depth_leres": "LeReS-DepthMapPreprocessor", "depth_metric3d": "Metric3D-DepthMapPreprocessor",
        "depth_meshgraphormer": "MeshGraphormer-DepthMapPreprocessor", "seg_ofcoco": "OneFormer-COCO-SemSegPreprocessor",
        "seg_ofade20k": "OneFormer-ADE20K-SemSegPreprocessor", "seg_ufade20k": "UniFormer-SemSegPreprocessor", "seg_animeface": "AnimeFace_SemSegPreprocessor",
        "shuffle": "ShufflePreprocessor", "teed": "TEEDPreprocessor", "color": "ColorPreprocessor", "sam": "SAMPreprocessor", "tile": "TilePreprocessor"
    }
    _PREPROCESSOR_TO_CNET_KEYWORD: Dict[str, str] = { # Populated by _initialize_preprocessors
        "canny": "canny", "canny_pyra": "canny", "lineart": "lineart", "lineart_anime": "lineart_anime", "lineart_manga": "lineart_manga",
        "lineart_any": "lineart", "scribble": "scribble", "scribble_xdog": "scribble", "scribble_pidi": "scribble", "scribble_hed": "scribble",
        "hed": "hed", "pidi": "pidi", "mlsd": "mlsd", "pose": "openpose", "openpose": "openpose", "dwpose": "openpose", "pose_dense": "densepose",
        "pose_animal": "animalpose", "normalmap_bae": "normalbae", "normalmap_dsine": "normal", "normalmap_midas": "normal", "depth": "depth",
        "depth_anything": "depth", "depth_anything_v2": "depth", "depth_anything_zoe": "depth", "depth_zoe": "depth", "depth_midas": "depth",
        "depth_leres": "depth", "depth_metric3d": "depth", "depth_meshgraphormer": "depth", "seg_ofcoco": "seg", "seg_ofade20k": "seg",
        "seg_ufade20k": "seg", "seg_animeface": "seg", "shuffle": "shuffle", "teed": "softedge", "color": "color", "sam": "inpaint", "tile": "tile"
    }

    @classmethod
    def _load_module(cls, module_path: str, module_name: Optional[str] = None) -> Optional[Any]:
        """ Dynamically loads a Python module from its file path. """
        # (Implementation unchanged from previous version)
        try:
            if module_name is None: module_name = os.path.basename(module_path);
            if os.path.isdir(module_path): module_path = os.path.join(module_path, "__init__.py")
            if not os.path.exists(module_path): logger.warning(f"Module path does not exist: {module_path}"); return None
            module_spec = importlib.util.spec_from_file_location(module_name, module_path)
            if module_spec is None: logger.warning(f"Could not create module spec for: {module_path}"); return None
            module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
            return module
        except Exception as e: logger.exception(f"Failed to load module '{module_name}' from {module_path}: {e}"); return None

    @classmethod
    def _initialize_preprocessors(cls: Type['FluxLoadControlNetPreprocessor']) -> None:
        """ Finds and loads the ControlNet preprocessor module and initializes necessary components. """
        # (Implementation unchanged from previous version)
        if cls._initialized: return
        logger.info(f"Initializing {cls.__name__}: Searching for preprocessor nodes...")
        module_path = None; preprocessors_dir_names = ["ControlNetPreprocessors", "comfyui_controlnet_aux"]
        try:
            custom_nodes_paths = folder_paths.get_folder_paths("custom_nodes");
            if not custom_nodes_paths: logger.warning("No custom node paths found."); return
            for custom_node_path in custom_nodes_paths:
                current_path = custom_node_path if not os.path.islink(custom_node_path) else os.readlink(custom_node_path)
                if not os.path.isdir(current_path): continue
                for module_dir in preprocessors_dir_names:
                    potential_path = os.path.join(current_path, module_dir)
                    if os.path.isdir(potential_path): module_path = os.path.abspath(potential_path); logger.info(f"Found potential preprocessor module at: {module_path}"); break
                if module_path: break
            if module_path is None: raise Exception("Could not find compatible ControlNet Preprocessor node pack.")
            module = cls._load_module(module_path)
            if module is None: raise Exception(f"Failed to load module from found path: {module_path}")
            logger.info(f"Successfully loaded ControlNet Preprocessor module from: {module_path}"); cls._preprocessor_module_path = module_path
            node_mappings: Optional[Dict] = getattr(module, "NODE_CLASS_MAPPINGS", None)
            preprocessor_options: Optional[List[str]] = getattr(module, "PREPROCESSOR_OPTIONS", None)
            if node_mappings is None or preprocessor_options is None: raise Exception(f"Module {module_path} missing expected attributes.")
            aio_preprocessor_class = node_mappings.get("AIO_Preprocessor", node_mappings.get("ControlNetPreprocessor", node_mappings.get("ImagePreprocessor", None)))
            if aio_preprocessor_class is None:
                found_aio = False
                for name, node_class in node_mappings.items():
                     if hasattr(node_class, "FUNCTION") and hasattr(node_class, "INPUT_TYPES"): logger.warning(f"AIO_Preprocessor not found, attempting to use '{name}'."); aio_preprocessor_class = node_class; found_aio = True; break
                if not found_aio: raise Exception("Could not find a suitable AIO Preprocessor class.")
            available = ["None"];
            for simple_name, class_name in cls._PREPROCESSORS_CLASS_MAP.items():
                if class_name in preprocessor_options: available.append(simple_name)
            cls._available_preprocessors = available; logger.info(f"Available preprocessors: {cls._available_preprocessors}")
            cls._aio_preprocessor_instance = aio_preprocessor_class(); logger.info(f"Instantiated AIO Preprocessor: {aio_preprocessor_class.__name__}")
            cls._initialized = True
        except Exception as e:
            cls._initialized = False; cls._available_preprocessors = ["None"]; cls._aio_preprocessor_instance = None
            logger.exception(f"{cls.__name__} initialization failed: {e}")

    @classmethod
    def INPUT_TYPES(cls: Type['FluxLoadControlNetPreprocessor']) -> Dict[str, Any]:
        """ Defines the input types for the node. """
        if not cls._initialized:
            cls._initialize_preprocessors()

        # --- Image Loading Inputs ---
        try:
            input_dir = folder_paths.get_input_directory()
            files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
            image_files = folder_paths.filter_files_content_types(files, ["image"])
            image_list = sorted(image_files)
        except Exception as e:
            logger.exception("Failed to get image file list for INPUT_TYPES.")
            image_list = ["Error: Could not list images"]

        return {
            "required": {
                # Preprocessor Inputs
                "preprocessor": (cls._available_preprocessors, {"tooltip": "Select the ControlNet preprocessor to apply."}),
                "sd_version": (["sd15", "sdxl"], {"tooltip": "Target Stable Diffusion version (for ControlNet name suggestion)."}),
                # Image Input (like LoadImage node)
                "image": (image_list, {"image_upload": True, "tooltip": "Select image file or upload."}),
            },
            "optional": {
                # Preprocessor Inputs
                "resolution": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64, "tooltip": "Target resolution for the preprocessor."}),
                "preprocessor_override": (["None"] + list(cls._PREPROCESSORS_CLASS_MAP.keys()), {"default": "None", "tooltip": "Manually select a preprocessor (overrides dropdown if not 'None')."}),
            },
        }

    # --- Instance Methods ---

    def _load_image_and_mask(self, image_filename: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Loads an image file, handles sequences (GIFs), normalizes,
        and extracts the alpha channel as a mask.

        Args:
            image_filename: The name of the image file in the input directory.

        Returns:
            A tuple containing (image_tensor, mask_tensor).
            Image tensor shape: [batch, height, width, channels]
            Mask tensor shape: [batch, height, width]

        Raises:
            FileNotFoundError: If the image file cannot be found.
            Exception: For other image loading or processing errors.
        """
        try:
            image_path = folder_paths.get_annotated_filepath(image_filename)
            if not os.path.exists(image_path):
                 raise FileNotFoundError(f"Image file not found at: {image_path}")

            logger.info(f"Loading image: {image_path}")
            # Use node_helpers for robust PIL opening
            img = node_helpers.pillow(Image.open, image_path)

            output_images = []
            output_masks = []
            w, h = None, None
            excluded_formats = ['MPO'] # Formats to not treat as sequences

            # Iterate through frames (handles single images and sequences like GIFs)
            for frame in ImageSequence.Iterator(img):
                # Handle EXIF orientation
                frame = node_helpers.pillow(ImageOps.exif_transpose, frame)

                # Handle different image modes
                if frame.mode == 'I': # Integer mode conversion
                    frame = frame.point(lambda i: i * (1 / 255.0)) # Normalize integer range
                image = frame.convert("RGB") # Ensure RGB

                # Check consistency for sequences
                if w is None: # First frame
                    w, h = image.size
                elif image.size[0] != w or image.size[1] != h:
                    logger.warning(f"Skipping frame in sequence {image_filename} due to inconsistent size.")
                    continue # Skip frames with inconsistent sizes

                # Convert image to tensor [1, H, W, C]
                image_np = np.array(image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(image_np)[None,]

                # Extract mask from alpha channel or transparency info
                mask = None
                if 'A' in frame.getbands():
                    mask_np = np.array(frame.getchannel('A')).astype(np.float32) / 255.0
                    mask = torch.from_numpy(mask_np)[None,] # Keep alpha format [1, H, W]
                    logger.debug("Extracted mask from alpha channel.")
                elif frame.mode == 'P' and 'transparency' in frame.info:
                     # Handle transparency in paletted images
                    try:
                         rgba_frame = frame.convert('RGBA')
                         mask_np = np.array(rgba_frame.getchannel('A')).astype(np.float32) / 255.0
                         mask = torch.from_numpy(mask_np)[None,]
                         logger.debug("Extracted mask from palette transparency.")
                    except Exception as e:
                        logger.warning(f"Could not extract mask from palette transparency: {e}. Using default mask.")

                if mask is None:
                    # Default mask (fully opaque - all ones) if no alpha/transparency
                    mask = torch.ones((1, h, w), dtype=torch.float32, device="cpu") # Matches image dimensions
                    logger.debug("No alpha/transparency found, using default opaque mask.")

                output_images.append(image_tensor)
                output_masks.append(mask)

            if not output_images:
                raise ValueError(f"Could not load any valid frames from image: {image_filename}")

            # Combine frames into a batch if multiple were loaded
            if len(output_images) > 1 and img.format not in excluded_formats:
                output_image_batch = torch.cat(output_images, dim=0)
                output_mask_batch = torch.cat(output_masks, dim=0)
                logger.info(f"Loaded image sequence with {len(output_images)} frames.")
            else:
                output_image_batch = output_images[0]
                output_mask_batch = output_masks[0]
                logger.info(f"Loaded single image frame.")

            return (output_image_batch, output_mask_batch)

        except FileNotFoundError as e:
            logger.error(f"Image loading failed: {e}")
            raise
        except Exception as e:
            logger.exception(f"Error processing image file {image_filename}: {e}")
            raise RuntimeError(f"Failed to load or process image '{image_filename}': {e}") from e


    def _apply_preprocessor(self, image: torch.Tensor, preprocessor_name: str, resolution: int) -> Optional[torch.Tensor]:
        """ Applies the selected preprocessor using the loaded AIO Preprocessor instance. """
        # (Implementation unchanged from previous version, calls AIO correctly)
        if not self._initialized or self._aio_preprocessor_instance is None: logger.error("Preprocessor system not initialized."); raise RuntimeError("Preprocessor system failed to initialize.")
        if preprocessor_name == "None": logger.info("Preprocessor is None, skipping."); return image
        if preprocessor_name not in self._PREPROCESSORS_CLASS_MAP: logger.error(f"Preprocessor name '{preprocessor_name}' not mapped."); raise ValueError(f"Invalid preprocessor name: {preprocessor_name}")
        preprocessor_cls_name = self._PREPROCESSORS_CLASS_MAP[preprocessor_name]
        if preprocessor_name not in self._available_preprocessors: logger.error(f"Mapped class '{preprocessor_cls_name}' not available in module."); raise ValueError(f"Preprocessor '{preprocessor_name}' not available.")
        logger.info(f"Applying preprocessor '{preprocessor_name}' (Class: {preprocessor_cls_name}) @ {resolution}px...")
        function_name = getattr(self._aio_preprocessor_instance, "FUNCTION", "execute")
        args = {"preprocessor": preprocessor_cls_name, "image": image, "resolution": resolution,} # Corrected args
        try:
            res = getattr(self._aio_preprocessor_instance, function_name)(**args)
            if isinstance(res, dict) and "result" in res: processed_image = res["result"][0]
            elif isinstance(res, tuple): processed_image = res[0]
            else: logger.warning(f"Unexpected result structure from AIO: {type(res)}."); processed_image = res
            logger.info(f"Preprocessor '{preprocessor_name}' applied successfully.")
            return processed_image
        except Exception as e: logger.exception(f"Error executing AIO preprocessor for '{preprocessor_name}': {e}"); raise RuntimeError(f"Preprocessor execution failed for {preprocessor_name}: {e}") from e

    def _detect_controlnet_name(self, preprocessor_name: str, sd_version: str) -> str:
        """ Suggests a common ControlNet model filename pattern. """
        # (Implementation unchanged)
        if preprocessor_name == "None" or preprocessor_name not in self._PREPROCESSOR_TO_CNET_KEYWORD: logger.info("No CNet name suggestion."); return "None"
        keyword = self._PREPROCESSOR_TO_CNET_KEYWORD[preprocessor_name]; version_tag = "xl" if sd_version == "sdxl" else "fp16"
        suggestion = f"control_{keyword}_{version_tag}"; logger.info(f"Suggested CNet pattern: '{suggestion}' for '{preprocessor_name}', SD '{sd_version}'.")
        return suggestion

    def execute(self, image: str, preprocessor: str, sd_version: str, resolution: int = 512, preprocessor_override: str = "None") -> Tuple[Any, Any, str]:
        """
        Main execution function: Loads image, applies preprocessor, returns results.

        Args:
            image: Filename of the image to load (from input/upload).
            preprocessor: Selected preprocessor simple name from dropdown.
            sd_version: Target SD version ("sd15" or "sdxl").
            resolution: Target resolution for preprocessing.
            preprocessor_override: Optional override for the preprocessor selection.

        Returns:
            Tuple containing (processed_image_tensor, mask_tensor, cnet_name_suggestion_string).
        """
        node_name = self.__class__.__name__
        logger.info(f"Executing node: {node_name}")

        # Ensure initialization is done
        if not self._initialized:
             logger.error("Node cannot execute because preprocessor initialization failed.")
             # Attempt to return dummy data matching expected types to avoid breaking workflow completely
             dummy_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32, device="cpu")
             dummy_mask = torch.ones((1, 64, 64), dtype=torch.float32, device="cpu")
             return (dummy_image, dummy_mask, "None")

        final_preprocessor = preprocessor
        # Handle override
        if preprocessor_override != "None":
            if preprocessor_override in self._available_preprocessors: logger.info(f"Overriding preprocessor with '{preprocessor_override}'."); final_preprocessor = preprocessor_override
            else: logger.warning(f"Override '{preprocessor_override}' not available. Using '{preprocessor}'.")

        try:
            # --- Step 1: Load Image and Mask ---
            logger.info("Step 1: Loading image...")
            loaded_image_tensor, loaded_mask_tensor = self._load_image_and_mask(image)

            # --- Step 2: Apply Preprocessor ---
            logger.info("Step 2: Applying preprocessor...")
            # Note: Preprocessor expects [B, H, W, C] usually
            processed_image = self._apply_preprocessor(loaded_image_tensor, final_preprocessor, resolution)

            # --- Step 3: Suggest ControlNet Name ---
            logger.info("Step 3: Suggesting ControlNet name...")
            control_net_name_suggestion = self._detect_controlnet_name(final_preprocessor, sd_version)

            logger.info(f"{node_name} execution completed successfully.")
            # Return processed image, original mask, and suggestion
            return (processed_image, loaded_mask_tensor, control_net_name_suggestion)

        except (FileNotFoundError, ValueError, RuntimeError) as e:
             logger.error(f"Execution failed in {node_name}: {e}", exc_info=True)
             raise # Re-raise for ComfyUI to handle
        except Exception as e:
             logger.exception(f"An unexpected critical error occurred in {node_name}: {e}")
             raise RuntimeError(f"Unexpected critical error in {node_name}: {e}") from e

# --- ComfyUI Registration ---
# Example (in __init__.py):
#
# from .your_combined_node_file import FluxLoadControlNetPreprocessor
#
# NODE_CLASS_MAPPINGS = {
#    "FluxLoadControlNetPreprocessor": FluxLoadControlNetPreprocessor
# }
#
# NODE_DISPLAY_NAME_MAPPINGS = {
#    "FluxLoadControlNetPreprocessor": "Load Image & Preprocess (Flux)"
# }