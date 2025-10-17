# /application/infrastructure/repositories/comfy_image_repository.py
__version__ = "1.1.1"
# 1.1.1 - Added full_path to the returned ImageSaveResult.
# 1.1.0 - Added logic to create yyyy-MM-dd subfolder when [root] is selected.
# 1.0.3 - Added logic to list and save to specific output subfolders.
# 1.0.2 - Improved metadata flag check for compatibility with different ComfyUI versions.
# 1.0.1 - Fixed missing 'Optional' type import.
# 1.0.0 - Concrete implementation of IImageRepository.

import os
import json
import logging
import datetime
import numpy as np
import torch
from typing import List, Optional
from PIL import Image as PILImage
from PIL.PngImagePlugin import PngInfo

import folder_paths
from domain.ports import IImageRepository
from domain.entities import Image, ImageSaveConfig, ImageSaveResult

try:
    from comfy import cli_args
except ImportError:
    class MockArgs:
        disable_save_metadata = False
    cli_args = MockArgs()

logger = logging.getLogger(__name__)

class ComfyImageRepository(IImageRepository):
    """
    Implements image saving using ComfyUI's backend and PIL.
    Handles discovery and saving to specific output subfolders.
    """
    ROOT_FOLDER_SENTINEL = "[root]"

    def __init__(self):
        self._output_dir = folder_paths.get_output_directory()
        self._type = "output"

    # --- MÉTODO REQUERIDO POR LA INTERFAZ ---
    def get_available_output_subfolders(self) -> List[str]:
        """See base class."""
        if not os.path.isdir(self._output_dir):
            return [self.ROOT_FOLDER_SENTINEL]
        try:
            subfolders = [d for d in os.listdir(self._output_dir) if os.path.isdir(os.path.join(self._output_dir, d))]
            return [self.ROOT_FOLDER_SENTINEL] + sorted(subfolders)
        except OSError:
            return [self.ROOT_FOLDER_SENTINEL]
    # ----------------------------------------

    def save_image(self, image: Image, config: ImageSaveConfig) -> ImageSaveResult:
        """See base class."""
        img_tensor: torch.Tensor = image.data
        img_height, img_width = img_tensor.shape[0:2]

        effective_subfolder = config.subfolder
        if effective_subfolder == self.ROOT_FOLDER_SENTINEL:
            effective_subfolder = datetime.date.today().strftime("%Y-%m-%d")

        output_path = self._output_dir
        if effective_subfolder:
            output_path = os.path.join(self._output_dir, effective_subfolder)
            os.makedirs(output_path, exist_ok=True)

        full_output_folder, filename, counter, subfolder, resolved_prefix = \
            folder_paths.get_save_image_path(config.filename_prefix, output_path, img_width, img_height)

        img_array = 255. * img_tensor.cpu().numpy()
        img_pil = PILImage.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
        metadata = self._create_metadata(config)
        file = f"{filename}_{counter:05}_.png"
        full_path = os.path.join(full_output_folder, file)
        img_pil.save(full_path, pnginfo=metadata, compress_level=config.compress_level)

        logger.info(f"Image saved to: {full_path}")
        
        final_subfolder = os.path.relpath(full_output_folder, self._output_dir)
        if final_subfolder == ".":
            final_subfolder = ""

        return ImageSaveResult(filename=file, subfolder=final_subfolder, full_path=full_path, type=self._type)

    def _create_metadata(self, config: ImageSaveConfig) -> Optional[PngInfo]:
        """Creates a PngInfo object from the provided metadata."""
        metadata_disabled = getattr(cli_args, 'disable_save_metadata', False) or \
                            getattr(cli_args, 'disable_metadata', False)
        if metadata_disabled:
            return None
        metadata = PngInfo()
        if config.metadata.prompt:
            metadata.add_text("prompt", json.dumps(config.metadata.prompt))
        if config.metadata.extra_info:
            for key, value in config.metadata.extra_info.items():
                metadata.add_text(key, json.dumps(value))
        return metadata