# -*- coding: utf-8 -*-
import torch
import logging
try:
    from domain.models import BufferConfig
    from infrastructure.persistence_adapter import FilePersistenceAdapter
except ImportError:
    from ..domain.models import BufferConfig
    from ..infrastructure.persistence_adapter import FilePersistenceAdapter

logger = logging.getLogger(__name__)

class ContinuityService:
    """
    Application Service:
    Manages the logic of switching between initial and persistent frames.
    """
    
    def __init__(self):
        self.persistence = FilePersistenceAdapter()

    def sync_image(self, config: BufferConfig, initial_image=None, sampler_last_image=None):
        # 1. Update disk if a new frame arrived from sampler
        if sampler_last_image is not None:
            self.persistence.save_frame(sampler_last_image)

        # 2. Logic for output
        if config.is_initial:
            if initial_image is not None:
                # Save initial to disk as well to prepare for next step
                self.persistence.save_frame(initial_image)
                return initial_image
            return torch.zeros((1, 480, 848, 3))
        
        # 3. Load from Disk
        persistent_frame = self.persistence.load_frame()
        if persistent_frame is not None:
            return persistent_frame
            
        # Fallback to zero if everything fails
        logger.error("[HEX] Continuity Error: No frame found, returning empty.")
        return torch.zeros((1, 480, 848, 3))
