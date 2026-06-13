# -*- coding: utf-8 -*-
import logging
try:
    from infrastructure.doc_adapter import hex_node_doc
except ImportError:
    from .infrastructure.doc_adapter import hex_node_doc

import os

from comfy.comfy_types import IO

from .domain.models import TextEncodingConfig
from .application.text_encoding_service import TextEncodingService
from .infrastructure.clip_encoding_adapter import ClipEncodingAdapter
from .infrastructure.error_adapter import hex_error_handler

logger = logging.getLogger(__name__)

_STYLES_PATH = os.path.join(os.path.dirname(__file__), "styles.csv")


@hex_node_doc
class FluxTextPromptHex:
    """
    [HEX] Flux Text Prompt.
    Codifica texto con hasta 4 estilos usando CLIP para modelos Flux.
    """

    @classmethod
    def INPUT_TYPES(cls):
        TextEncodingService.load_styles(_STYLES_PATH)
        style_names = TextEncodingService.get_style_names()

        return {
            "required": {
                # --- SECTION: PROMPT ---
                "section_prompt": ("STRING", {"default": "TEXT & STYLES"}),
                "text":    (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "Prompt principal."}),
                "style1":  (style_names, {"default": "No Style"}),
                "style2":  (style_names, {"default": "No Style"}),
                "style3":  (style_names, {"default": "No Style"}),
                "style4":  (style_names, {"default": "No Style"}),

                # --- SECTION: ENCODING ---
                "section_encoding": ("STRING", {"default": "ENCODING"}),
                "clip":     ("CLIP",   {"tooltip": "Modelo CLIP."}),
                "guidance": (IO.FLOAT, {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Guidance scale."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION     = "execute"
    CATEGORY     = "flux_collection_advanced/hex"

    @hex_error_handler
    def execute(self, **kwargs):
        config = TextEncodingConfig(
            text     = kwargs.get("text", ""),
            style1   = kwargs.get("style1", "No Style"),
            style2   = kwargs.get("style2", "No Style"),
            style3   = kwargs.get("style3", "No Style"),
            style4   = kwargs.get("style4", "No Style"),
            guidance = kwargs.get("guidance", 3.5),
        )

        service = TextEncodingService(clip_adapter=ClipEncodingAdapter())
        return (service.encode(config, kwargs.get("clip")),)
