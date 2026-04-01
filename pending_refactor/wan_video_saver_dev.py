# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import logging
import folder_paths
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WanVideoSaver_Dev:
    """
    [DEV] Wan Video Saver v1.1:
    Homemade video encoder with smart folder organization.
    Organizes by Date/Videos or Custom path.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "Scene"}),
                "custom_path": ("STRING", {"default": "", "placeholder": "Leave empty for auto-date folder"}),
                "fps": ("INT", {"default": 25, "min": 1, "max": 120}),
                "index": ("INT", {"default": 0, "min": 0, "max": 9999}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_video"
    OUTPUT_NODE = True
    CATEGORY = "flux_collection_advanced/_dev"

    def save_video(self, images, filename_prefix, custom_path, fps, index):
        try:
            import imageio
        except ImportError:
            raise RuntimeError("imageio library is required for Wan Video Saver.")

        # 1. Determine Target Directory
        base_output = folder_paths.get_output_directory()
        
        if custom_path.strip():
            # Use custom path (relative to output or absolute)
            if os.path.isabs(custom_path):
                target_dir = custom_path
            else:
                target_dir = os.path.join(base_output, custom_path)
        else:
            # Auto-Date Organization: output/YYYY-MM-DD/videos/
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            target_dir = os.path.join(base_output, today, "videos")

        # 2. Create directory structure if missing
        os.makedirs(target_dir, exist_ok=True)

        # 3. Format Filename: Scene_0001_HHMMSS.mp4
        timestamp = datetime.datetime.now().strftime("%H%M%S")
        filename = f"{filename_prefix}_{index:04}_{timestamp}.mp4"
        full_path = os.path.join(target_dir, filename)

        logger.info(f"[DEV] Saver: Exporting to {full_path}...")

        # 4. Process Frames
        video_data = []
        for frame in images:
            f_np = (frame.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            video_data.append(f_np)

        # 5. Write File
        imageio.mimwrite(full_path, video_data, fps=fps, quality=8, macro_block_size=16)

        logger.info(f"[DEV] Export Successful: {full_path}")
        return {}

# Registered via __init__.py
