# -*- coding: utf-8 -*-
"""
[HEX] Image Load & Encode
Nodo que combina Load Image y VAE Encode en un solo paso para agilizar workflows de Img2Img.
"""
import logging
import os
import numpy as np
import torch
from PIL import Image, ImageOps
import folder_paths

try:
    from infrastructure.error_adapter import hex_error_handler
except ImportError:
    from .infrastructure.error_adapter import hex_error_handler

logger = logging.getLogger(__name__)


class ImageLoadEncodeHex:
    """
    Combina la carga de imagen desde el disco con la decodificación VAE (si se provee un VAE).
    Genera simultáneamente la imagen y el latente.
    """
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        try:
            files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        except Exception:
            files = []
            
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
            },
            "optional": {
                "vae": ("VAE", {"tooltip": "Conectar para autogenerar el LATENT"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT", "MASK")
    RETURN_NAMES = ("image", "latent", "mask")
    FUNCTION = "execute"
    CATEGORY = "flux_collection_advanced/utils"

    @hex_error_handler
    def execute(self, image, vae=None):
        logger.info(f"[HEX] Cargando imagen: {image}")
        image_path = folder_paths.get_annotated_filepath(image)
        
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        image_rgb = i.convert("RGB")
        
        image_np = np.array(image_rgb).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]
        
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - mask
            mask_tensor = torch.from_numpy(mask)[None,]
        else:
            mask_tensor = torch.zeros((1, 64, 64), dtype=torch.float32)

        latent = None
        if vae is not None:
            logger.info("[HEX] Aplicando VAE Encode a la imagen cargada...")
            # comfy.sd.VAE.encode devuelve un tensor de latentes directamente
            latent = vae.encode(image_tensor[:,:,:,:3])
            
        return (
            image_tensor, 
            {"samples": latent} if latent is not None else None, 
            mask_tensor
        )
