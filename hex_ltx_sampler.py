# Task-Source: T#33
# -*- coding: utf-8 -*-
import logging
try:
    from infrastructure.doc_adapter import hex_node_doc
except ImportError:
    from .infrastructure.doc_adapter import hex_node_doc

import os
import tempfile

import numpy as np
import torch
from PIL import Image

try:
    from domain.models import LTXGenerationSettings
    from application.ltx_video_generation_service import LTXVideoGenerationService
    from infrastructure.ltx_pipeline_adapter import LTXPipelineAdapter
    from infrastructure.error_adapter import ErrorLoggingAdapter, hex_error_handler
except ImportError:
    from .domain.models import LTXGenerationSettings
    from .application.ltx_video_generation_service import LTXVideoGenerationService
    from .infrastructure.ltx_pipeline_adapter import LTXPipelineAdapter
    from .infrastructure.error_adapter import ErrorLoggingAdapter, hex_error_handler

logger = logging.getLogger(__name__)


@hex_node_doc
class LTXSamplerHex:
    """
    [HEX] LTX Video 2.3 Sampler.
    Genera video a partir de un pipeline LTX cargado, soportando texto puro (T2V)
    o condicionamiento de imagen opcional (I2V).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "section_sampling": ("STRING", {"default": "SAMPLING PARAMETERS"}),
                "ltx_pipeline": ("LTX_PIPELINE",),
                "prompt": ("STRING", {"multiline": True, "default": "a slow camera pan across a realistic forest portrait, high definition"}),
                "width": ("INT", {"default": 768, "min": 64, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 32}),
                "num_frames": ("INT", {"default": 97, "min": 9, "max": 257, "step": 8}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            },
            "optional": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LTX_AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "execute"
    CATEGORY = "flux_collection_advanced/hex_ltx"

    @hex_error_handler
    def execute(self, **kwargs):
        pipeline = kwargs.get("ltx_pipeline")
        prompt = kwargs.get("prompt", "")
        width = kwargs.get("width", 768)
        height = kwargs.get("height", 512)
        num_frames = kwargs.get("num_frames", 97)
        frame_rate = kwargs.get("frame_rate", 24.0)
        seed = kwargs.get("seed", 0)

        image = kwargs.get("image", None)
        strength = kwargs.get("strength", 1.0)

        # Preparar imagen condicional (I2V) si se proporciona
        image_path = ""
        temp_file = None
        if image is not None:
            frame_tensor = image[0]  # [H, W, C]
            np_frame = (frame_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            pil_image = Image.fromarray(np_frame)
            pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)

            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, f"ltx_hex_cond_{seed}.png")
            pil_image.save(temp_file)
            image_path = temp_file
            logger.debug(f"[HEX-LTX] Imagen condicional guardada en: {temp_file}")

        try:
            settings = LTXGenerationSettings(
                prompt=prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                frame_rate=frame_rate,
                seed=seed,
                strength=strength,
                image_path=image_path,
            )

            service = LTXVideoGenerationService(
                pipeline_adapter=LTXPipelineAdapter(),
                error_adapter=ErrorLoggingAdapter(),
            )

            video_output, audio_data = service.generate(pipeline, settings)
            return (video_output, audio_data)

        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    logger.warning(f"[HEX-LTX] No se pudo eliminar imagen condicional temporal: {e}")
