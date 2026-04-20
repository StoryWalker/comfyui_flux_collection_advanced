# -*- coding: utf-8 -*-
import os
import logging
import torch
from typing import Any, Dict, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WanPromptEncoder_Dev:
    """
    [DEV] Wan Prompt Encoder:
    Encodes text from the Sequencer into Conditioning using CLIP and optional styles.
    """
    
    _cached_styles = {}
    _styles_loaded = False

    @classmethod
    def _load_styles(cls):
        if not cls._styles_loaded:
            styles_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "styles.csv")
            if os.path.exists(styles_path):
                try:
                    with open(styles_path, "r", encoding="utf-8") as f:
                        import re
                        lines = f.readlines()[1:]
                        csv_regex = re.compile(r',(?=(?:[^"]*"[^"]*")*[^"]*$)')
                        for line in lines:
                            if not line.strip(): continue
                            parts = [p.strip().strip('"') for p in csv_regex.split(line)]
                            if len(parts) >= 3:
                                cls._cached_styles[parts[0]] = [parts[1], parts[2]]
                    cls._styles_loaded = True
                except Exception as e:
                    # Task-Source: T#3
                    logger.warning(f"[DEV] Error al cargar styles.csv: {e}")
            if not cls._cached_styles:
                cls._cached_styles = {"No Style": ["", ""]}
                cls._styles_loaded = True

    @classmethod
    def INPUT_TYPES(cls):
        cls._load_styles()
        style_names = list(cls._cached_styles.keys())
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"forceInput": True}),
                "style_1": (style_names, {"default": "No Style"}),
                "style_2": (style_names, {"default": "No Style"}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("positive", "negative",)
    FUNCTION = "encode"
    CATEGORY = "flux_collection_advanced/_dev"

    def _get_full_prompt(self, base_text, s1, s2, is_positive=True):
        idx = 0 if is_positive else 1
        prompts = [base_text]
        for s in [s1, s2]:
            if s in self._cached_styles:
                style_text = self._cached_styles[s][idx]
                if style_text: prompts.append(style_text)
        return ", ".join([p for p in prompts if p.strip()])

    def encode(self, clip, text, style_1, style_2):
        internal_neg = "blurry, low quality, distorted, static, oversaturated, grayness, ugly, text, watermark"
        
        full_pos = self._get_full_prompt(text, style_1, style_2, True)
        full_neg = self._get_full_prompt(internal_neg, style_1, style_2, False)

        # Positive
        tokens_p = clip.tokenize(full_pos)
        cond_p, pooled_p = clip.encode_from_tokens(tokens_p, return_pooled=True)
        positive = [[cond_p, {"pooled_output": pooled_p}]]

        # Negative
        tokens_n = clip.tokenize(full_neg)
        cond_n, pooled_n = clip.encode_from_tokens(tokens_n, return_pooled=True)
        negative = [[cond_n, {"pooled_output": pooled_n}]]

        return (positive, negative)

# Registered via __init__.py
