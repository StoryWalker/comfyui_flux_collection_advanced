# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import logging
import datetime
import folder_paths
from typing import List

logger = logging.getLogger(__name__)

class ImageIOSaver:
    """
    Infrastructure Adapter for saving video files using imageio.
    Handles ComfyUI path resolution and file system operations.
    """
    
    @staticmethod
    def resolve_full_path(filename_prefix: str, index: int, custom_path: str) -> str:
        """
        Resolves the final absolute path for the video file.
        Encapsulates ComfyUI folder_paths and OS logic.
        """
        base_output = folder_paths.get_output_directory()
        
        # 1. Determine Target Directory
        if custom_path.strip():
            if os.path.isabs(custom_path):
                target_dir = custom_path
            else:
                target_dir = os.path.join(base_output, custom_path)
        else:
            # Auto-Date Organization: output/YYYY-MM-DD/videos/
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            target_dir = os.path.join(base_output, today, "videos")

        # 2. Ensure directory exists
        os.makedirs(target_dir, exist_ok=True)

        # 3. Format Filename
        timestamp = datetime.datetime.now().strftime("%H%M%S")
        filename = f"{filename_prefix}_{index:04}_{timestamp}.mp4"
        
        return os.path.join(target_dir, filename)

    @staticmethod
    def save_mp4(full_path: str, frames: torch.Tensor, fps: int):
        try:
            import imageio
        except ImportError:
            raise RuntimeError("imageio not found.")

        # Process frames into uint8 list
        video_data = []
        for frame in frames:
            f_np = (frame.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            video_data.append(f_np)

        # Write file
        imageio.mimwrite(full_path, video_data, fps=fps, quality=8, macro_block_size=16)
        logger.info(f"[HEX] Video saved successfully to {full_path}")
