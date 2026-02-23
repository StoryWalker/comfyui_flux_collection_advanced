# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from PIL import Image
import folder_paths
import logging

logger = logging.getLogger(__name__)

class FilePersistenceAdapter:
    """
    Infrastructure Adapter for persistent image storage.
    Used to bridge executions in infinite loops (Continuity).
    """
    
    def __init__(self, filename="wan_continuity_buffer.png"):
        self.filename = filename
        self.path = os.path.join(folder_paths.get_temp_directory(), self.filename)

    def save_frame(self, image_tensor: torch.Tensor):
        """ Saves a single frame tensor [1, H, W, C] to disk for next iteration """
        if image_tensor is None:
            return
            
        try:
            # Comfy images are [B, H, W, C], we take first batch [0]
            i = 255. * image_tensor[0].cpu().numpy()
            img_pil = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img_pil.save(self.path)
            logger.info(f"[HEX] Persistence: Frame saved for next cycle to {self.path}")
        except Exception as e:
            logger.error(f"Failed to save persistence frame: {e}")

    def load_frame(self) -> torch.Tensor:
        """ Loads the persistent frame from disk as a tensor """
        if not os.path.exists(self.path):
            logger.warning(f"[HEX] Persistence: No frame found at {self.path}")
            return None
        
        try:
            img_pil = Image.open(self.path).convert("RGB")
            img_tensor = torch.from_numpy(np.array(img_pil).astype(np.float32) / 255.0).unsqueeze(0)
            logger.info(f"[HEX] Persistence: Frame loaded from disk.")
            return img_tensor
        except Exception as e:
            logger.error(f"Failed to load persistence frame: {e}")
            return None

    def exists(self) -> bool:
        """ Checks if the buffer file exists on disk """
        return os.path.exists(self.path)
