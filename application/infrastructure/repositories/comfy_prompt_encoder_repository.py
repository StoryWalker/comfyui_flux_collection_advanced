# /application/infrastructure/repositories/comfy_prompt_encoder_repository.py
__version__ = "1.0.0"
# 1.0.0 - Concrete implementation of IPromptEncoderRepository.

import os
import re
import logging
from typing import Dict, List
from functools import lru_cache

import folder_paths
import node_helpers
from domain.ports import IPromptEncoderRepository
from domain.entities import PromptConfig, Conditioning, Style

logger = logging.getLogger(__name__)

class ComfyPromptEncoderRepository(IPromptEncoderRepository):
    """
    Concrete repository that implements prompt encoding by:
    1. Reading styles from a CSV file within the custom node directory.
    2. Using ComfyUI's CLIP and node_helpers to perform tokenization and encoding.
    """
    _STYLES_FILENAME = "styles.csv"

    @lru_cache(maxsize=1)
    def get_available_styles(self) -> Dict[str, Style]:
        """
        Loads style definitions from a CSV file. Uses caching to avoid re-reading the file.
        See base class for details.
        """
        # We need to find the path to the custom node directory itself.
        # This assumes this file is at .../infrastructure/repositories/
        current_dir = os.path.dirname(__file__)
        # Navigate up to the custom node's root directory
        custom_node_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
        styles_path = os.path.join(custom_node_dir, self._STYLES_FILENAME)
        
        styles: Dict[str, Style] = {}
        logger.info(f"Attempting to load styles from: {styles_path}")
        if not os.path.exists(styles_path):
            logger.error(f"Styles file not found at: {styles_path}")
            return {"Error: styles.csv not found": Style("Error", "", "")}

        try:
            with open(styles_path, "r", encoding="utf-8") as f:
                lines = f.readlines()[1:]  # Skip header
                csv_regex = re.compile(r',(?=(?:[^"]*"[^"]*")*[^"]*$)')
                for i, line in enumerate(lines):
                    if not line.strip(): continue
                    parts = [p.strip().strip('"') for p in csv_regex.split(line)]
                    if len(parts) >= 3:
                        style = Style(name=parts[0], positive_prompt=parts[1], negative_prompt=parts[2])
                        if style.name:
                            styles[style.name] = style
                    else:
                        logger.warning(f"Skipping malformed row {i+2} in styles.csv")
            logger.info(f"Successfully loaded {len(styles)} styles.")
            return styles
        except Exception as e:
            logger.exception(f"Failed to read or process styles file {styles_path}: {e}")
            return {"Error loading styles.csv": Style("Error", "", "")}
            
    def encode_prompt(self, config: PromptConfig, styles: Dict[str, Style]) -> Conditioning:
        """See base class."""
        if config.clip_model is None:
            raise ValueError("CLIP model provided to repository cannot be None.")

        # 1. Construct the full prompt text
        style_prompts: List[str] = []
        for style_name in config.style_names:
            if style_name in styles and styles[style_name].positive_prompt:
                style_prompts.append(styles[style_name].positive_prompt)
        
        full_prompt = config.base_text.strip()
        if style_prompts:
            separator = ', ' if full_prompt else ''
            full_prompt += separator + ', '.join(style_prompts)
        
        logger.debug(f"Constructed prompt: {full_prompt}")
        if not full_prompt:
            logger.warning("Encoding an empty prompt.")

        # 2. Use ComfyUI's CLIP object to tokenize and encode
        tokens = config.clip_model.tokenize(full_prompt)
        conditioning_data = config.clip_model.encode_from_tokens_scheduled(tokens)

        # 3. Apply guidance using node_helpers
        final_conditioning_data = node_helpers.conditioning_set_values(
            conditioning_data, {"guidance": config.guidance}
        )

        return Conditioning(data=final_conditioning_data)