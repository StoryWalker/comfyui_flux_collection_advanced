# Task-Source: T#32
# -*- coding: utf-8 -*-
import logging
import os

try:
    from domain.models import LTXPipelineConfig
    from infrastructure.ltx_pipeline_adapter import LTXPipelineAdapter
    from infrastructure.error_adapter import ErrorLoggingAdapter, hex_error_handler
    from ltx_nodes import scan_ltx_files, resolve_ltx_path
except ImportError:
    from .domain.models import LTXPipelineConfig
    from .infrastructure.ltx_pipeline_adapter import LTXPipelineAdapter
    from .infrastructure.error_adapter import ErrorLoggingAdapter, hex_error_handler
    from .ltx_nodes import scan_ltx_files, resolve_ltx_path

logger = logging.getLogger(__name__)


class LTXLoaderHex:
    """
    [HEX] LTX Video 2.3 Unified Loader.
    Carga pipelines LTX Fast (SafeTensors) o Distilled GGUF desde carpetas nativas de ComfyUI.
    """

    _CHECKPOINT_CATEGORIES = ["diffusion_models", "checkpoints", "unet"]
    _GGUF_CATEGORIES = ["diffusion_models", "checkpoints", "unet"]
    _GEMMA_CATEGORIES = ["clip", "text_encoders"]
    _UPSCALER_CATEGORIES = ["latent_upscale_models", "upscale_models"]
    _VAE_CATEGORIES = ["vae"]
    _CONNECTOR_CATEGORIES = ["text_encoders", "clip"]

    @classmethod
    def INPUT_TYPES(cls):
        # Fast pipeline files
        fast_ckpt_list = scan_ltx_files(cls._CHECKPOINT_CATEGORIES, ".safetensors", "ltx-2.3-22b-distilled.safetensors")
        # GGUF pipeline files
        gguf_ckpt_list = scan_ltx_files(cls._GGUF_CATEGORIES, ".gguf", "ltx-2.3-22b-distilled-Q4_K_M.gguf")
        # Shared resources
        gemma_list = scan_ltx_files(cls._GEMMA_CATEGORIES, "dir", "gemma-3-12b-it-qat-q4_0-unquantized")
        upscaler_list = scan_ltx_files(cls._UPSCALER_CATEGORIES, ".safetensors", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors")
        vae_list = scan_ltx_files(cls._VAE_CATEGORIES, ".safetensors", "ltx-2.3-22b-dev_video_vae.safetensors")
        audio_vae_list = scan_ltx_files(cls._VAE_CATEGORIES, ".safetensors", "ltx-2.3-22b-dev_audio_vae.safetensors")
        connector_list = scan_ltx_files(cls._CONNECTOR_CATEGORIES, ".safetensors", "ltx-2.3-22b-dev_embeddings_connectors.safetensors")

        return {
            "required": {
                "section_pipeline": ("STRING", {"default": "PIPELINE TYPE"}),
                "pipeline_type": (["fast", "distilled_gguf"], {"default": "fast"}),
                "device": (["cuda", "cpu"], {"default": "cuda"}),

                "section_models": ("STRING", {"default": "MODEL FILES"}),
                "checkpoint_fast": (fast_ckpt_list,),
                "checkpoint_gguf": (gguf_ckpt_list,),
                "gemma_directory": (gemma_list,),
                "spatial_upscaler": (upscaler_list,),

                "section_gguf_extra": ("STRING", {"default": "GGUF EXTRAS (only for distilled_gguf)"}),
                "vae_video": (vae_list,),
                "audio_vae": (audio_vae_list,),
                "connector": (connector_list,),
            }
        }

    RETURN_TYPES = ("LTX_PIPELINE",)
    RETURN_NAMES = ("ltx_pipeline",)
    FUNCTION = "execute"
    CATEGORY = "flux_collection_advanced/hex_ltx"
    DESCRIPTION = (
        "[HEX] LTX Video 2.3 Unified Loader.\nCarga pipelines LTX Fast (SafeTensors) o Distilled GGUF desde carpetas nativas de ComfyUI.\n\n"
        "This documentation was AI-generated. If you find any errors or have suggestions for "
        "improvement, please feel free to contribute! Edit on GitHub."
    )

    @hex_error_handler
    def execute(self, **kwargs):
        pipeline_type = kwargs.get("pipeline_type", "fast")
        device = kwargs.get("device", "cuda")

        if pipeline_type == "fast":
            checkpoint = kwargs.get("checkpoint_fast")
            vae_video = ""
            audio_vae = ""
            connector = ""
        else:
            checkpoint = kwargs.get("checkpoint_gguf")
            vae_video = kwargs.get("vae_video", "")
            audio_vae = kwargs.get("audio_vae", "")
            connector = kwargs.get("connector", "")

        gemma_directory = kwargs.get("gemma_directory")
        spatial_upscaler = kwargs.get("spatial_upscaler")

        # Resolver rutas
        if pipeline_type == "fast":
            ckpt_path = resolve_ltx_path(checkpoint, self._CHECKPOINT_CATEGORIES)
        else:
            ckpt_path = resolve_ltx_path(checkpoint, self._GGUF_CATEGORIES)

        gemma_path = resolve_ltx_path(gemma_directory, self._GEMMA_CATEGORIES)
        upscaler_path = resolve_ltx_path(spatial_upscaler, self._UPSCALER_CATEGORIES)
        vae_video_path = resolve_ltx_path(vae_video, self._VAE_CATEGORIES) if vae_video and vae_video != "None" else ""
        audio_vae_path = resolve_ltx_path(audio_vae, self._VAE_CATEGORIES) if audio_vae and audio_vae != "None" else ""
        connector_path = resolve_ltx_path(connector, self._CONNECTOR_CATEGORIES) if connector and connector != "None" else ""

        # Validar existencia
        required = [
            ("checkpoint", ckpt_path),
            ("gemma_directory", gemma_path),
            ("spatial_upscaler", upscaler_path),
        ]
        if pipeline_type == "distilled_gguf":
            required.extend([
                ("vae_video", vae_video_path),
                ("audio_vae", audio_vae_path),
                ("connector", connector_path),
            ])

        missing_resources = []
        for name, p in required:
            if not p or not os.path.exists(p):
                missing_resources.append((name, p))

        if missing_resources:
            msg_lines = [f"[LTX LoaderHex Error] Faltan los siguientes recursos:"]
            for name, p in missing_resources:
                msg_lines.append(f"  • {name}: {p}")
            msg_lines.append("")
            msg_lines.append("Consulta ltx_models.md para la lista completa de modelos requeridos.")
            msg_lines.append("Descarga los modelos LTX 2.3 y colócalos en las carpetas de ComfyUI indicadas.")
            raise FileNotFoundError("\n".join(msg_lines))

        config = LTXPipelineConfig(
            checkpoint_path=ckpt_path,
            gemma_root=gemma_path,
            upsampler_path=upscaler_path,
            vae_video_path=vae_video_path,
            audio_vae_path=audio_vae_path,
            connector_path=connector_path,
            pipeline_type=pipeline_type,
            device=device,
        )

        adapter = LTXPipelineAdapter()
        pipeline = adapter.load_pipeline(config)
        logger.info(f"[HEX-LTX] Pipeline cargado exitosamente: {pipeline_type}")
        return (pipeline,)
