# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
import logging
import folder_paths
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WanContinuityBuffer_Dev:
    """
    [DEV] Wan Continuity Buffer v1.1:
    Smart switch for infinite storytelling.
    - initial_image: The starting photo.
    - sampler_last_image: The feedback frame from the sampler.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mode": (["Initial Frame", "Continue from Disk", "Reset/Static", "Save/Load"], {"default": "Initial Frame"}),
            },
            "optional": {
                "initial_image": ("IMAGE", {"tooltip": "Connect your 'Load Image' node here."}),
                "sampler_last_image": ("IMAGE", {"tooltip": "Connect 'last_image' from the sampler here."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_out",)
    FUNCTION = "sync_buffer"
    CATEGORY = "flux_collection_advanced/_dev"

    def sync_buffer(self, mode, initial_image=None, sampler_last_image=None):
        buffer_path = os.path.join(folder_paths.get_temp_directory(), "wan_continuity_buffer.png")
        
        # 1. Mode: Initial Frame / Reset (Start of the story)
        if mode in ["Initial Frame", "Reset/Static"]:
            if initial_image is not None:
                # We also save it to disk to prepare for the transition to 'Continue'
                i = 255. * initial_image[0].cpu().numpy()
                img_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                img_pil.save(buffer_path)
                logger.info("[DEV] Buffer: Initialized with starting image.")
                return (initial_image,)
            else:
                logger.warning("[DEV] Buffer: Mode set to Initial but no image connected.")
                return (torch.zeros((1, 480, 848, 3)),)

        # 2. Mode: Continue from Disk / Save/Load (Story loop)
        else:
            # First, check if we just received a new frame to save for the NEXT iteration
            if sampler_last_image is not None:
                logger.info("[DEV] Buffer: Saving frame from sampler output...")
                i = 255. * sampler_last_image[0].cpu().numpy()
                img_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
                img_pil.save(buffer_path)
            
            # Then, always return what is currently on disk
            if os.path.exists(buffer_path):
                logger.info("[DEV] Buffer: Delivering persistent frame from disk.")
                img_pil = Image.open(buffer_path).convert("RGB")
                img_tensor = torch.from_numpy(np.array(img_pil).astype(np.float32) / 255.0).unsqueeze(0)
                return (img_tensor,)
            else:
                logger.error("[DEV] Buffer: No persistent frame found on disk!")
                return (torch.zeros((1, 480, 848, 3)),)

class WanPromptSequencer_Dev:
    """
    [DEV] Wan Prompt Sequencer:
    Cycles through a multiline text prompt, one line per execution.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompts": ("STRING", {"multiline": True, "default": "Scene 1\nScene 2\nScene 3"}),
                "current_index": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "INT",)
    RETURN_NAMES = ("current_prompt", "next_index",)
    FUNCTION = "get_prompt"
    CATEGORY = "flux_collection_advanced/_dev"

    def get_prompt(self, prompts, current_index):
        lines = [l.strip() for l in prompts.split("\n") if l.strip()]
        if not lines:
            return ("", 0)
        index = current_index % len(lines)
        selected = lines[index]
        logger.info(f"[DEV] Sequencer: Selected prompt index {index}")
        return (selected, current_index + 1)

# Registered via __init__.py
