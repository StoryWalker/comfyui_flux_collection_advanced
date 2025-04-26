# -*- coding: utf-8 -*-
import os
import logging
import importlib.util
from typing import Any, Dict, List, Tuple, Optional, Type

# Necessary third-party imports
import torch
import folder_paths
import nodes # For inheritance

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- RENAMED CLASS ---
class FluxLoadControlNetPreprocessor(nodes.ComfyNodeABC):
    """
    A ComfyUI node that applies a selected ControlNet preprocessor to an image
    and suggests a corresponding ControlNet model name pattern.
    It dynamically loads preprocessors from compatible custom node packs
    (like comfyui_controlnet_aux or ControlNetPreprocessors).
    """

    # --- Node Metadata ---
    FUNCTION = "execute"
    CATEGORY = "Art Venture/Loaders" # Keeping original category unless asked to change
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("IMAGE", "CNET_NAME_SUGGESTION")
    OUTPUT_NODE = False

    # --- Class Attributes for Preprocessor Management ---
    _initialized: bool = False
    _available_preprocessors: List[str] = ["None"]
    _aio_preprocessor_instance: Optional[Any] = None
    _preprocessor_module_path: Optional[str] = None

    # Mapping from simple names to actual preprocessor class names
    _PREPROCESSORS_CLASS_MAP: Dict[str, str] = {
        "canny": "CannyEdgePreprocessor", "canny_pyra": "PyraCannyPreprocessor",
        "lineart": "LineArtPreprocessor", "lineart_anime": "AnimeLineArtPreprocessor",
        "lineart_manga": "Manga2Anime_LineArt_Preprocessor", "lineart_any": "AnyLineArtPreprocessor_aux",
        "scribble": "ScribblePreprocessor", "scribble_xdog": "Scribble_XDoG_Preprocessor",
        "scribble_pidi": "Scribble_PiDiNet_Preprocessor", "scribble_hed": "FakeScribblePreprocessor",
        "hed": "HEDPreprocessor", "pidi": "PiDiNetPreprocessor", "mlsd": "M-LSDPreprocessor",
        "pose": "DWPreprocessor", "openpose": "OpenposePreprocessor", "dwpose": "DWPreprocessor",
        "pose_dense": "DensePosePreprocessor", "pose_animal": "AnimalPosePreprocessor",
        "normalmap_bae": "BAE-NormalMapPreprocessor", "normalmap_dsine": "DSINE-NormalMapPreprocessor",
        "normalmap_midas": "MiDaS-NormalMapPreprocessor",
        "depth": "DepthAnythingV2Preprocessor", "depth_anything": "DepthAnythingPreprocessor", "depth_anything_v2": "DepthAnythingV2Preprocessor",
        "depth_anything_zoe": "Zoe_DepthAnythingPreprocessor", "depth_zoe": "Zoe-DepthMapPreprocessor",
        "depth_midas": "MiDaS-DepthMapPreprocessor", "depth_leres": "LeReS-DepthMapPreprocessor",
        "depth_metric3d": "Metric3D-DepthMapPreprocessor", "depth_meshgraphormer": "MeshGraphormer-DepthMapPreprocessor",
        "seg_ofcoco": "OneFormer-COCO-SemSegPreprocessor", "seg_ofade20k": "OneFormer-ADE20K-SemSegPreprocessor",
        "seg_ufade20k": "UniFormer-SemSegPreprocessor", "seg_animeface": "AnimeFace_SemSegPreprocessor",
        "shuffle": "ShufflePreprocessor", "teed": "TEEDPreprocessor", "color": "ColorPreprocessor",
        "sam": "SAMPreprocessor", "tile": "TilePreprocessor"
    }

    # Simple mapping for suggesting ControlNet file names/keywords
    _PREPROCESSOR_TO_CNET_KEYWORD: Dict[str, str] = {
        "canny": "canny", "canny_pyra": "canny",
        "lineart": "lineart", "lineart_anime": "lineart_anime",
        "lineart_manga": "lineart_manga", "lineart_any": "lineart",
        "scribble": "scribble", "scribble_xdog": "scribble",
        "scribble_pidi": "scribble", "scribble_hed": "scribble",
        "hed": "hed", "pidi": "pidi", "mlsd": "mlsd",
        "pose": "openpose", "openpose": "openpose", "dwpose": "openpose",
        "pose_dense": "densepose", "pose_animal": "animalpose",
        "normalmap_bae": "normalbae", "normalmap_dsine": "normal", "normalmap_midas": "normal",
        "depth": "depth", "depth_anything": "depth", "depth_anything_v2": "depth",
        "depth_anything_zoe": "depth", "depth_zoe": "depth", "depth_midas": "depth",
        "depth_leres": "depth", "depth_metric3d": "depth", "depth_meshgraphormer": "depth",
        "seg_ofcoco": "seg", "seg_ofade20k": "seg", "seg_ufade20k": "seg", "seg_animeface": "seg",
        "shuffle": "shuffle", "teed": "softedge",
        "color": "color", "sam": "inpaint", "tile": "tile"
    }

    @classmethod
    def _load_module(cls, module_path: str, module_name: Optional[str] = None) -> Optional[Any]:
        """ Dynamically loads a Python module from its file path. """
        # (Implementation unchanged)
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
    def _initialize_preprocessors(cls: Type['FluxLoadControlNetPreprocessor']) -> None: # Updated type hint
        """ Finds and loads the ControlNet preprocessor module and initializes necessary components. """
        # (Implementation unchanged)
        if cls._initialized: return
        logger.info("Initializing FluxLoadControlNetPreprocessor: Searching for preprocessor nodes...")
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
            logger.exception(f"FluxLoadControlNetPreprocessor initialization failed: {e}") # Updated log message class name

    @classmethod
    def INPUT_TYPES(cls: Type['FluxLoadControlNetPreprocessor']) -> Dict[str, Any]: # Updated type hint
        """ Defines the input types for the node. """
        if not cls._initialized:
            cls._initialize_preprocessors()
        return {
            "required": {
                "image": ("IMAGE",),
                "preprocessor": (cls._available_preprocessors, {"tooltip": "Select the ControlNet preprocessor to apply."}),
                "sd_version": (["sd15", "sdxl"], {"tooltip": "Target Stable Diffusion version (for ControlNet name suggestion)."}),
            },
            "optional": {
                "resolution": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 64, "tooltip": "Target resolution for the preprocessor."}),
                "preprocessor_override": (["None"] + list(cls._PREPROCESSORS_CLASS_MAP.keys()), {"default": "None", "tooltip": "Manually select a preprocessor (overrides dropdown if not 'None')."}),
            },
        }

    # --- Instance Methods ---

    def _apply_preprocessor(self, image: torch.Tensor, preprocessor_name: str, resolution: int) -> Optional[torch.Tensor]:
        """
        Applies the selected preprocessor using the loaded AIO Preprocessor instance.
        """
        # (Implementation unchanged, uses corrected args call)
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

    def execute(self, image: torch.Tensor, preprocessor: str, sd_version: str, resolution: int = 512, preprocessor_override: str = "None") -> Tuple[Any, str]:
        """ Main execution function for the node. """
        # (Implementation unchanged)
        node_name = self.__class__.__name__ # Uses the new class name
        logger.info(f"Executing node: {node_name}")
        if not self._initialized: logger.error("Node cannot execute: preprocessor initialization failed."); return (image, "None")
        final_preprocessor = preprocessor
        if preprocessor_override != "None":
            if preprocessor_override in self._available_preprocessors: logger.info(f"Overriding with '{preprocessor_override}'."); final_preprocessor = preprocessor_override
            else: logger.warning(f"Override '{preprocessor_override}' not available. Using '{preprocessor}'.")
        try:
            processed_image = self._apply_preprocessor(image, final_preprocessor, resolution)
            control_net_name_suggestion = self._detect_controlnet_name(final_preprocessor, sd_version)
            logger.info(f"{node_name} execution completed successfully.")
            return (processed_image, control_net_name_suggestion)
        except (ValueError, RuntimeError) as e: logger.error(f"Execution failed in {node_name}: {e}", exc_info=True); raise
        except Exception as e: logger.exception(f"Unexpected critical error in {node_name}: {e}"); raise RuntimeError(f"Unexpected critical error in {node_name}: {e}") from e

# --- ComfyUI Registration ---
# Example (in __init__.py):
#
# from .your_preprocessor_node_file import FluxLoadControlNetPreprocessor # Use the new class name
#
# NODE_CLASS_MAPPINGS = {
#    "FluxLoadControlNetPreprocessor": FluxLoadControlNetPreprocessor # Use the new class name
# }
#
# NODE_DISPLAY_NAME_MAPPINGS = {
#    "FluxLoadControlNetPreprocessor": "Flux ControlNet Preprocessor" # Update display name if desired
# }