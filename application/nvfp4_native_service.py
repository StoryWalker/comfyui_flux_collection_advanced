# -*- coding: utf-8 -*-
import logging
from typing import Any, Dict

try:
    from domain.models import FluxModelConfig
    from infrastructure.nvfp4_native_adapter import NVFP4NativeAdapter
    from infrastructure.nvfp4_model_adapter import NVFP4ModelAdapter
except ImportError:
    from ..domain.models import FluxModelConfig
    from ..infrastructure.nvfp4_native_adapter import NVFP4NativeAdapter
    from ..infrastructure.nvfp4_model_adapter import NVFP4ModelAdapter

logger = logging.getLogger(__name__)

class NVFP4NativeService:
    def __init__(self, native_adapter=None, std_adapter=None):
        self.native_adapter = native_adapter or NVFP4NativeAdapter()
        self.std_adapter = std_adapter or NVFP4ModelAdapter()

    def load_stack(self, config: FluxModelConfig) -> Dict[str, Any]:
        logger.info(f"[NVFP4 Native Service] Iniciando carga de stack completo: {config.unet_name}")
        
        unet = self.native_adapter.load_unet(config.unet_name)
        clip = self.std_adapter.load_clip(config)
        vae = self.std_adapter.load_vae(config.vae_name)
        
        return {
            "model": unet,
            "clip": clip,
            "vae": vae
        }
