# -*- coding: utf-8 -*-
"""
Application Service for NVFP4 Image Generation.
Orchestrates model loading, text encoding, and sampling for NVFP4 pipelines.
"""
import logging
from typing import Any, Dict, Optional, Tuple

try:
    from domain.models import FluxModelConfig, FluxSamplerConfig, TextEncodingConfig
    from infrastructure.nvfp4_model_adapter import NVFP4ModelAdapter
    from infrastructure.clip_encoding_adapter import ClipEncodingAdapter
    from infrastructure.image_sampler_adapter import FluxSamplerAdapter
except ImportError:
    from ..domain.models import FluxModelConfig, FluxSamplerConfig, TextEncodingConfig
    from ..infrastructure.nvfp4_model_adapter import NVFP4ModelAdapter
    from ..infrastructure.clip_encoding_adapter import ClipEncodingAdapter
    from ..infrastructure.image_sampler_adapter import FluxSamplerAdapter

logger = logging.getLogger(__name__)


class NVFP4ImageGenerationService:
    """
    Service that coordinates the full NVFP4 image generation pipeline:
      1. Load model stack (UNET + CLIP + VAE)
      2. Encode text prompt via CLIP
      3. Run sampling and VAE decode
    """

    def __init__(
        self,
        model_adapter: NVFP4ModelAdapter = None,
        clip_adapter: ClipEncodingAdapter = None,
        sampler_adapter: FluxSamplerAdapter = None,
    ):
        self.model_adapter = model_adapter or NVFP4ModelAdapter()
        self.clip_adapter = clip_adapter or ClipEncodingAdapter()
        self.sampler_adapter = sampler_adapter or FluxSamplerAdapter()

    def load_stack(self, config: FluxModelConfig) -> Dict[str, Any]:
        """Load the full model stack."""
        logger.info(f"[NVFP4] Loading stack: {config.unet_name}")
        model = self.model_adapter.load_unet(config)
        if config.base_type in ("flux", "flux2"):
            self.model_adapter.apply_flux_sampling(model)
        clip = self.model_adapter.load_clip(config)
        vae = self.model_adapter.load_vae(config.vae_name)
        return {"model": model, "clip": clip, "vae": vae}

    def encode_prompt(self, config: TextEncodingConfig, clip: Any) -> Any:
        """Encode text prompt to conditioning."""
        from application.text_encoding_service import TextEncodingService
        service = TextEncodingService(clip_adapter=self.clip_adapter)
        return service.encode(config, clip)

    def sample(
        self,
        sampler_config: FluxSamplerConfig,
        model: Any,
        positive: Any,
        vae: Any,
        latent_opt: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Run sampling and VAE decode."""
        from application.image_sampler_service import FluxSamplerService
        service = FluxSamplerService(adapter=self.sampler_adapter)
        return service.generate(sampler_config, model, positive, vae, latent_opt)
