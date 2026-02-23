# -*- coding: utf-8 -*-
import torch
import comfy.utils
import node_helpers
from typing import Any, Tuple

class TorchImageAdapter:
    """
    Infrastructure Adapter for PyTorch and ComfyUI Image Utils.
    Translates Domain requests into tensor operations.
    """
    
    @staticmethod
    def resize_for_wan(image: torch.Tensor, width: int, height: int, method: str, crop: str) -> torch.Tensor:
        """ Specialized Wan 2.2 Resizer """
        # ComfyUI expects [B, C, H, W] for upscale
        img_in = image.movedim(-1, 1)
        img_resized = comfy.utils.common_upscale(img_in, width, height, method, crop)
        return img_resized.movedim(1, -1) # Return [B, H, W, C]

    @staticmethod
    def get_center_crop(image: torch.Tensor) -> torch.Tensor:
        """ Precise square center crop for CLIP Vision stability """
        h, w = image.shape[1:3]
        length = min(h, w)
        y, x = (h - length) // 2, (w - length) // 2
        return image[:, y:y+length, x:x+length, :]

    @staticmethod
    def encode_vision(clip_vision: Any, image: torch.Tensor) -> Any:
        """ Standard CLIP Vision encoding wrapper """
        # We assume image is already center-cropped by the adapter
        return clip_vision.encode_image(image, crop=False)

    @staticmethod
    def build_5d_context(vae: Any, frames: int, width: int, height: int, resized_img: torch.Tensor, device: torch.device):
        """ Specialized 5D Latent Construction for Wan 2.2 """
        latent_t = ((frames - 1) // 4) + 1
        
        # Initial latent [zeros]
        latent = torch.zeros([1, 16, latent_t, height // 8, width // 8], device=device)
        
        # Sequence: Only Frame 0 is our reference image, rest is padding
        sequence = torch.ones((frames, height, width, 3), device=device, dtype=resized_img.dtype) * 0.5
        sequence[0] = resized_img[0]
        
        # VAE Encode full sequence for 'concat_latent_image'
        concat_img = vae.encode(sequence[:, :, :, :3])
        
        # Mask: Anchor first block
        concat_mask = torch.ones((1, 1, latent_t, height // 8, width // 8), device=device)
        concat_mask[:, :, :1] = 0.0 
        
        return latent, concat_img, concat_mask
