# Task-Source: T#27
# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import logging
try:
    from infrastructure.doc_adapter import hex_node_doc
except ImportError:
    from .infrastructure.doc_adapter import hex_node_doc

import folder_paths
import datetime
from .infrastructure.error_adapter import hex_error_handler

logger = logging.getLogger(__name__)


@hex_node_doc
class WanVideoSaverDevHex:
    """
    [HEX] Wan Video Saver (DEV):
    Exportador de video alternativo con organización por carpetas y codificación directa via imageio.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "section_saver": ("STRING", {"default": "WAN VIDEO SAVER (DEV)"}),
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "Scene"}),
                "custom_path": ("STRING", {"default": "", "placeholder": "Leave empty for auto-date folder"}),
                "fps": ("INT", {"default": 25, "min": 1, "max": 120}),
                "index": ("INT", {"default": 0, "min": 0, "max": 9999}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "flux_collection_advanced/hex"

    @hex_error_handler
    def execute(self, **kwargs):
        images = kwargs["images"]
        filename_prefix = kwargs.get("filename_prefix", "Scene")
        custom_path = kwargs.get("custom_path", "")
        fps = kwargs.get("fps", 25)
        index = kwargs.get("index", 0)

        try:
            import imageio
        except ImportError:
            raise RuntimeError("imageio library is required for Wan Video Saver (DEV).")

        base_output = folder_paths.get_output_directory()
        
        if custom_path.strip():
            if os.path.isabs(custom_path):
                target_dir = custom_path
            else:
                target_dir = os.path.join(base_output, custom_path)
        else:
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            target_dir = os.path.join(base_output, today, "videos")

        os.makedirs(target_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%H%M%S")
        filename = f"{filename_prefix}_{index:04}_{timestamp}.mp4"
        full_path = os.path.join(target_dir, filename)

        logger.info(f"[HEX] Saver DEV: Exporting to {full_path}...")

        video_data = []
        for frame in images:
            f_np = (frame.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            video_data.append(f_np)

        imageio.mimwrite(full_path, video_data, fps=fps, quality=8, macro_block_size=16)

        logger.info(f"[HEX] Export DEV Successful: {full_path}")
        return {}
