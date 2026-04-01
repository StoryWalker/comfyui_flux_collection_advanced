# -*- coding: utf-8 -*-
import logging
import os
import folder_paths
try:
    from domain.models import PromptSequence
except ImportError:
    from ..domain.models import PromptSequence

logger = logging.getLogger(__name__)

class PromptSequencerService:
    """
    Application Service:
    Handles the logic for cycling through prompts and formatting them.
    Supports index persistence to disk.
    """
    def __init__(self):
        self.index_file = os.path.join(folder_paths.get_temp_directory(), "wan_prompt_index.txt")

    def _read_persistent_index(self) -> int:
        # Task-Source: T#2
        if os.path.exists(self.index_file):
            try:
                with open(self.index_file, "r") as f:
                    return int(f.read().strip())
            except (ValueError, OSError) as e:
                logger.warning(f"[HEX] Prompt Service: No se pudo leer el indice persistente, reiniciando a 0: {e}")
                return 0
        return 0

    def _save_persistent_index(self, index: int):
        with open(self.index_file, "w") as f:
            f.write(str(index))

    def get_next_prompt(self, raw_text: str, manual_index: int, mode: str) -> tuple:
        """
        Selected prompt based on mode: 'Manual' or 'Auto-Increment'.
        """
        if mode == "Auto-Increment":
            idx = self._read_persistent_index()
        else:
            idx = manual_index

        sequence = PromptSequence(raw_text=raw_text, current_index=idx)
        selected_line = sequence.get_current_line()
        
        # Increment and save for next run if in auto mode
        next_val = sequence.next_index
        if mode == "Auto-Increment":
            self._save_persistent_index(next_val)
            logger.info(f"[HEX] Prompt Service: Auto-incremented to {next_val}")
        
        return selected_line, next_val
