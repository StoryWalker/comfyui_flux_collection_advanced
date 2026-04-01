# -*- coding: utf-8 -*-
import logging
from typing import Any, Dict, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WanIndexBridge_Dev:
    """
    [DEV] Wan Index Bridge:
    A simple bridge to allow the Sequencer to loop its index back 
    to itself, bypassing ComfyUI's direct-loop restriction.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "index_in": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("index_out",)
    FUNCTION = "bridge"
    CATEGORY = "flux_collection_advanced/_dev"

    def bridge(self, index_in):
        return (index_in,)

# Registered via __init__.py
