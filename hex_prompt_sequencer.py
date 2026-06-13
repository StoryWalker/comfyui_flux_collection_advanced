# -*- coding: utf-8 -*-
import logging
try:
    from infrastructure.doc_adapter import hex_node_doc
except ImportError:
    from .infrastructure.doc_adapter import hex_node_doc

import time
from .application.prompt_service import PromptSequencerService
from .infrastructure.error_adapter import hex_error_handler

logger = logging.getLogger(__name__)

@hex_node_doc
class PromptSequencerHex:
    """
    [HEX] v3.2.0 Robust Prompt Sequencer.
    Ensures each queued job uses the NEXT index.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "section_guion": ("STRING", {"default": "SEQUENCER SETTINGS"}),
                "prompts": ("STRING", {"multiline": True, "default": "Scene 1: Forest\nScene 2: Lake\nScene 3: Mountain"}),
                "mode": (["Auto-Increment", "Manual"], {"default": "Auto-Increment"}),
                "current_index": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "INT",)
    RETURN_NAMES = ("CURRENT_PROMPT", "NEXT_INDEX",)
    FUNCTION = "execute"
    CATEGORY = "flux_collection_advanced/hex"

    # Forces ComfyUI to re-execute and update index on every Queue press
    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()

    @hex_error_handler
    def execute(self, **kwargs):
        prompts = kwargs.get("prompts", "")
        mode = kwargs.get("mode", "Auto-Increment")
        current_index = kwargs.get("current_index", 0)
        
        service = PromptSequencerService()
        selected_prompt, next_idx = service.get_next_prompt(prompts, current_index, mode)
        
        return (selected_prompt, next_idx)

# Registered via __init__.py
