import os
import re
import logging
from typing import Any, Dict, List, Tuple, Type, Optional # Keep Any for type hints

# Third-party imports
import node_helpers
import folder_paths
from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict # Keep necessary imports

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FluxTextPrompt(ComfyNodeABC):
    """
    A ComfyUI node designed to encode a textual prompt using a specified CLIP model.

    Processes text, applies styles, tokenizes/encodes with CLIP, and applies guidance.
    Adheres to SOLID principles and provides robust error handling.
    """

    # ComfyUI Node Identification
    CATEGORY = "flux_collection_advanced"
    # *** Use the STRING IDENTIFIER for the return type ***
    RETURN_TYPES = ("CONDITIONING",)
    OUTPUT_TOOLTIPS = ("The conditioning tensor derived from the encoded text.",)
    FUNCTION = "encode_text_with_styles"
    DESCRIPTION = "Encodes text with styles using CLIP for FLUX models."

    # --- Class Level Attributes ---
    _cached_styles: Dict[str, List[str]] = {}
    _styles_loaded: bool = False
    _error_loading_styles: Optional[str] = None
    _STYLES_FILENAME = "styles.csv"
    _DEFAULT_ERROR_STYLE = {"Error loading styles.csv": ["", ""]}

    @classmethod
    def load_styles_csv(cls, styles_path: str) -> Dict[str, List[str]]:
        """ Loads style definitions from a CSV file. (See previous implementation) """
        styles: Dict[str, List[str]] = {}
        if not os.path.exists(styles_path):
            raise FileNotFoundError(f"Styles file not found at: {styles_path}")
        try:
            with open(styles_path, "r", encoding="utf-8") as f:
                lines = f.readlines()[1:]
                csv_regex = re.compile(r',(?=(?:[^"]*"[^"]*")*[^"]*$)')
                for i, line in enumerate(lines):
                    if not line.strip(): continue
                    try:
                        parts = [part.strip().strip('"') for part in csv_regex.split(line)]
                        if len(parts) >= 3:
                            style_name, pos_prompt, neg_prompt = parts[0], parts[1], parts[2]
                            if style_name: styles[style_name] = [pos_prompt, neg_prompt]
                            else: logger.warning(f"Skipping style with empty name in row {i+2} of {styles_path}")
                        else: logger.warning(f"Skipping malformed row {i+2} in {styles_path}: Expected 3+ columns, got {len(parts)}. Line: '{line.strip()}'")
                    except Exception as parse_error: logger.error(f"Error parsing row {i+2} in {styles_path}: {parse_error}. Line: '{line.strip()}'")
        except Exception as e:
            logger.exception(f"Failed to read or process styles file {styles_path}: {e}")
            raise
        if not styles: logger.warning(f"No valid styles were loaded from {styles_path}.")
        return styles

    @classmethod
    def _ensure_styles_loaded(cls) -> None:
        """ Internal method to load styles if not already loaded. (See previous implementation) """
        if not cls._styles_loaded:
            styles_file_path = os.path.join(folder_paths.base_path, cls._STYLES_FILENAME)
            try:
                logger.info(f"Attempting to load styles from: {styles_file_path}")
                cls._cached_styles = cls.load_styles_csv(styles_file_path)
                cls._styles_loaded = True
                cls._error_loading_styles = None
                if not cls._cached_styles:
                     cls._error_loading_styles = f"No styles found or loaded from {styles_file_path}. Using default error entry."
                     cls._cached_styles = cls._DEFAULT_ERROR_STYLE.copy()
                     logger.warning(cls._error_loading_styles)
                else: logger.info(f"Successfully loaded {len(cls._cached_styles)} styles.")
            except FileNotFoundError:
                cls._error_loading_styles = f"ERROR: {cls._STYLES_FILENAME} not found in {folder_paths.base_path}."
                logger.error(cls._error_loading_styles)
                cls._cached_styles = cls._DEFAULT_ERROR_STYLE.copy(); cls._styles_loaded = True
            except Exception as e:
                cls._error_loading_styles = f"Error loading {cls._STYLES_FILENAME}: {e}"
                logger.exception(cls._error_loading_styles)
                cls._cached_styles = cls._DEFAULT_ERROR_STYLE.copy(); cls._styles_loaded = True

    @classmethod
    def INPUT_TYPES(cls: Type['FluxTextPrompt']) -> InputTypeDict:
        """ Defines the required input types for the node. """
        cls._ensure_styles_loaded()
        style_names = list(cls._cached_styles.keys())
        if not style_names: style_names = ["Error: No Styles Available"]
        clip_type_identifier = "CLIP" # String identifier for CLIP type

        return {
            "required": {
                "text": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "Main text prompt."}),
                "clip": (clip_type_identifier, {"tooltip": "CLIP model instance."}),
                "style1": (style_names, {"tooltip": "First style."}),
                "style2": (style_names, {"tooltip": "Second style."}),
                "style3": (style_names, {"tooltip": "Third style."}),
                "guidance": (IO.FLOAT, {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Guidance scale."})
            }
        }

    def _validate_inputs(self, clip: Any, text: str, style1: str, style2: str, style3: str) -> None:
        """ Validates core inputs. """
        if clip is None: raise ValueError("Invalid CLIP input: Received None.")
        if not isinstance(text, str): raise ValueError(f"Invalid text input: Expected string, got {type(text).__name__}")
        if self._error_loading_styles and style1 not in self._DEFAULT_ERROR_STYLE:
             logger.warning(f"Style loading failed. Style '{style1}' might be invalid.")
        elif not self._error_loading_styles:
             for style_name in [style1, style2, style3]:
                 if style_name not in self._cached_styles:
                      raise ValueError(f"Selected style '{style_name}' not found.")

    def _get_style_prompt(self, style_name: str) -> str:
        """ Retrieves the positive prompt string for a style. """
        if style_name in self._DEFAULT_ERROR_STYLE: return ""
        if style_name in self._cached_styles:
            try:
                style_data = self._cached_styles[style_name]
                if isinstance(style_data, list) and len(style_data) > 0:
                    return str(style_data[0])
                else: logger.warning(f"Cached style '{style_name}' malformed."); return ""
            except Exception as e: logger.warning(f"Error accessing style '{style_name}': {e}."); return ""
        else: logger.warning(f"Style '{style_name}' not found."); return ""

    def _construct_full_prompt(self, text: str, style1: str, style2: str, style3: str) -> str:
        """ Combines base text with style prompts. """
        prompts = [self._get_style_prompt(s) for s in [style1, style2, style3]]
        style_parts = [p for p in prompts if p and p.strip()]
        full_prompt = text.strip()
        if style_parts:
            separator = ', ' if full_prompt else ''
            full_prompt += separator + ', '.join(style_parts)
        logger.debug(f"Constructed prompt: {full_prompt}")
        return full_prompt

    def _tokenize_text(self, clip: Any, text: str) -> Any:
        """ Tokenizes text using the CLIP model. """
        if not hasattr(clip, 'tokenize'): raise RuntimeError("CLIP object missing 'tokenize' method.")
        try: return clip.tokenize(text)
        except Exception as e: logger.exception("Tokenization error."); raise RuntimeError(f"Tokenization failed: {e}")

    def _encode_tokens(self, clip: Any, tokens: Any) -> Any:
        """ Encodes tokens using the CLIP model. """
        if not hasattr(clip, 'encode_from_tokens_scheduled'): raise RuntimeError("CLIP object missing 'encode_from_tokens_scheduled'.")
        try: return clip.encode_from_tokens_scheduled(tokens)
        except Exception as e: logger.exception("Encoding error."); raise RuntimeError(f"Encoding failed: {e}")

    def _apply_guidance(self, conditioning: Any, guidance: float) -> Any:
        """ Applies guidance using node_helpers. """
        if not hasattr(node_helpers, 'conditioning_set_values'): raise RuntimeError("node_helpers.conditioning_set_values not found.")
        try: return node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})
        except Exception as e: logger.exception("Guidance application error."); raise RuntimeError(f"Applying guidance failed: {e}")

    # Keep internal type hints using Any where specific types weren't importable
    def encode_text_with_styles(self, clip: Any, text: str, style1: str, style2: str, style3: str, guidance: float) -> Tuple[Any,]:
        """ Main execution function. """
        node_name = self.__class__.__name__
        logger.info(f"Executing node: {node_name}")
        try:
            self._validate_inputs(clip, text, style1, style2, style3)
            full_prompt = self._construct_full_prompt(text, style1, style2, style3)
            if not full_prompt: logger.warning("Empty prompt after combining text/styles.")
            tokens = self._tokenize_text(clip, full_prompt)
            conditioning = self._encode_tokens(clip, tokens)
            final_conditioning = self._apply_guidance(conditioning, guidance)
            logger.info(f"{node_name} execution successful.")
            # The actual returned object's type should match "CONDITIONING"
            return (final_conditioning,)
        except (ValueError, RuntimeError) as e:
            logger.error(f"Execution failed in {node_name}: {e}"); raise
        except Exception as e:
            logger.exception(f"Unexpected critical error in {node_name}: {e}")
            raise RuntimeError(f"Unexpected critical error in {node_name}: {e}") from e

# --- ComfyUI Registration ---
# Ensure this class is registered in your __init__.py
# NODE_CLASS_MAPPINGS = { "FluxTextPrompt": FluxTextPrompt }
# NODE_DISPLAY_NAME_MAPPINGS = { "FluxTextPrompt": "Flux Text Prompt Styler" }