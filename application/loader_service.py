import logging
import torch
import comfy.model_management
try:
    from domain.models import WanModelConfig, ImageLoadConfig
    from infrastructure.model_adapter import ComfyModelAdapter
    from infrastructure.image_io_adapter import ImageIOAdapter
except ImportError:
    from ..domain.models import WanModelConfig, ImageLoadConfig
    from ..infrastructure.model_adapter import ComfyModelAdapter
    from ..infrastructure.image_io_adapter import ImageIOAdapter

logger = logging.getLogger(__name__)

class ImageLoaderService:
    """
    Application Service:
    Coordinates image loading and basic preprocessing.
    """
    
    def __init__(self):
        self.io_adapter = ImageIOAdapter()

    def load_image(self, config: ImageLoadConfig) -> torch.Tensor:
        """
        Loads a single image from disk using the domain config.
        """
        logger.info(f"[HEX] Image Loader Service: Loading {config.image_path}")
        return self.io_adapter.load_single_image(config.image_path)

class ModelLoaderService:
    """
    Application Service:
    Orchestrates the entire Wan 2.2 model loading process.
    """
    
    def __init__(self):
        self.adapter = ComfyModelAdapter()

    def load_full_wan_stack(self, config: WanModelConfig) -> dict:
        """
        Coordinates the sequential loading and patching of Wan components.
        Ensures strict order and clean VRAM to avoid dimension mismatches.
        """
        logger.info(f"[HEX] Loader Service: Performing memory cleanup...")
        comfy.model_management.cleanup_models()
        
        logger.info(f"[HEX] Loader Service: Starting Wan 2.2 Stack Load (CLIP Mode: {config.t5_optimization})")

        # 1. Load CLIP First (Most sensitive to dimension errors)
        clip = self.adapter.load_clip_wan(config.clip_name, config.t5_optimization, config.t5_layers)

        # 2. Load Base UNETs
        m_high = self.adapter.load_unet(config.model_high_name, config.weight_dtype)
        m_low = self.adapter.load_unet(config.model_low_name, config.weight_dtype)

        # 2. Apply LoRA
        if config.lora_name != "None":
            logger.info(f"[HEX] Applying LoRA: {config.lora_name}")
            m_high, clip = self.adapter.apply_lora_robust(m_high, clip, config.lora_name, config.lora_strength)
            m_low, _ = self.adapter.apply_lora_robust(m_low, None, config.lora_name, config.lora_strength)

        # 3. Apply Sampling Shift
        logger.info(f"[HEX] Applying Shift: {config.sampling_shift}")
        m_high = self.adapter.apply_sampling_shift(m_high, config.sampling_shift)
        m_low = self.adapter.apply_sampling_shift(m_low, config.sampling_shift)

        # 4. Load VAE and Vision (Optional)
        vae, cv = None, None
        if config.vae_name or config.clip_vision_name:
            vae, cv = self.adapter.load_vae_and_vision(config.vae_name, config.clip_vision_name)

        return {
            "model_high": m_high,
            "model_low": m_low,
            "clip": clip,
            "vae": vae,
            "clip_vision": cv
        }
