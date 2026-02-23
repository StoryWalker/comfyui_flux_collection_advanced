# -*- coding: utf-8 -*-
import os
import torch
import numpy as np
from PIL import Image, ImageOps
import folder_paths
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

class ImageIOAdapter:
    """
    Infrastructure Adapter for generic image loading from disk.
    Converts filesystem images into ComfyUI-compatible tensors.
    """
    
    @staticmethod
    def load_single_image(image_name: str) -> torch.Tensor:
        """ Loads a single image from ComfyUI's input directory using absolute paths """
        input_dir = folder_paths.get_input_directory()
        image_path = os.path.join(input_dir, image_name)
        
        if not os.path.exists(image_path):
            # Try Comfy's annotated path as fallback
            image_path = folder_paths.get_annotated_filepath(image_name)

        img = Image.open(image_path)
        img = ImageOps.exif_transpose(img)
        image = img.convert("RGB")
        
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        
        return image

    @staticmethod
    def load_styles_csv() -> dict:
        """
        Loads styles from styles.csv file.
        Returns a dict: {name: (positive, negative)}
        """
        import csv
        styles = {"No Style": ("", "")}
        # styles.csv is in the root of the custom node
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_path, "styles.csv")
        
        if not os.path.exists(csv_path):
            logger.warning(f"styles.csv not found at {csv_path}")
            return styles

        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader) # Skip header
                for row in reader:
                    if len(row) >= 3:
                        styles[row[0]] = (row[1], row[2])
            logger.info(f"[HEX] Loaded {len(styles)} styles from CSV.")
        except Exception as e:
            logger.error(f"Error loading styles: {e}")
            
        return styles

    @staticmethod
    def get_input_image_list() -> List[str]:
        """ 
        Returns a list of all images in the input directory.
        Uses manual OS listing for robustness against ComfyUI init state.
        """
        placeholder = "[ Select an Image ]"
        try:
            input_dir = folder_paths.get_input_directory()
            if not os.path.exists(input_dir):
                return [placeholder, "Input folder missing"]

            extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
            images = [f for f in os.listdir(input_dir) 
                     if os.path.isfile(os.path.join(input_dir, f)) 
                     and f.lower().endswith(extensions)]
            
            if not images:
                return [placeholder, "No images found"]
            
            return [placeholder] + sorted(images)
        except Exception as e:
            logger.error(f"Manual scan failed: {e}")
            return [placeholder, "Error reading folder"]
