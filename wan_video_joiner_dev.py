# -*- coding: utf-8 -*-
import torch
import logging
import comfy.utils
from typing import Any, Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WanVideoJoiner_Dev:
    """
    [DEV] Wan Video Joiner:
    Concatenates multiple video segments (image sequences) into a single continuous video.
    Ensures all segments match in resolution before merging.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_1": ("IMAGE", {"tooltip": "First video segment."}),
                "video_2": ("IMAGE", {"tooltip": "Second video segment."}),
            },
            "optional": {
                "video_3": ("IMAGE", {"tooltip": "Optional third segment."}),
                "video_4": ("IMAGE", {"tooltip": "Optional fourth segment."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("combined_video",)
    FUNCTION = "join_segments"
    CATEGORY = "flux_collection_advanced/_dev"

    def join_segments(self, video_1, video_2, video_3=None, video_4=None):
        logger.info("[DEV] Video Joiner: Concatenating segments...")
        
        segments = [video_1, video_2]
        if video_3 is not None: segments.append(video_3)
        if video_4 is not None: segments.append(video_4)

        # 1. Determine base resolution from the first segment [F, H, W, C]
        target_h, target_w = segments[0].shape[1], segments[0].shape[2]
        
        processed_segments = []
        
        for i, seg in enumerate(segments):
            # 2. Ensure each segment is 4D [Frames, H, W, C]
            if len(seg.shape) == 3: # Single image
                seg = seg.unsqueeze(0)
            elif len(seg.shape) == 5: # Batch dimension exists
                seg = seg.squeeze(0)

            # 3. Resize if resolution mismatch
            current_h, current_w = seg.shape[1], seg.shape[2]
            if current_h != target_h or current_w != target_w:
                logger.warning(f"[DEV] Segment {i+1} resolution mismatch ({current_w}x{current_h}). Resizing to {target_w}x{target_h}...")
                # common_upscale expects BCHW
                seg_bchw = seg.movedim(-1, 1)
                seg_resized = comfy.utils.common_upscale(seg_bchw, target_w, target_h, "bilinear", "center")
                seg = seg_resized.movedim(1, -1)
            
            processed_segments.append(seg)

        # 4. Concatenate along the frame dimension (dim 0)
        combined = torch.cat(processed_segments, dim=0)
        
        logger.info(f"[DEV] Success! Combined video has {combined.shape[0]} total frames.")
        return (combined,)

# Registered via __init__.py
