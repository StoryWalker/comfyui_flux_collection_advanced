# Task-Source: T#34
# -*- coding: utf-8 -*-
import logging
try:
    from infrastructure.doc_adapter import hex_node_doc
except ImportError:
    from .infrastructure.doc_adapter import hex_node_doc

import os

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    from domain.models import VideoExportManifest
    from application.export_service import ExportVideoService
    from infrastructure.error_adapter import hex_error_handler
    from ltx_backend import save_audio_wav, mux_video_audio_with_ffmpeg
except ImportError:
    from .domain.models import VideoExportManifest
    from .application.export_service import ExportVideoService
    from .infrastructure.error_adapter import hex_error_handler
    from .ltx_backend import save_audio_wav, mux_video_audio_with_ffmpeg

logger = logging.getLogger(__name__)


@hex_node_doc
class LTXVideoSaverHex:
    """
    [HEX] LTX Video 2.3 Saver.
    Guarda frames de video en MP4 y retorna el payload UI para previsualización
    interactiva via ltx_video_preview.js.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "section_export": ("STRING", {"default": "EXPORT CONFIGURATION"}),
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "LTX_Video"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
            },
            "optional": {
                "audio": ("LTX_AUDIO",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "flux_collection_advanced/hex_ltx"

    @hex_error_handler
    def execute(self, **kwargs):
        images = kwargs.get("images")
        if images is None:
            return {}

        filename_prefix = kwargs.get("filename_prefix", "LTX_Video")
        fps = kwargs.get("fps", 24)
        audio = kwargs.get("audio", None)

        manifest = VideoExportManifest(
            filename_prefix=filename_prefix,
            index=0,
            fps=fps,
            custom_path="",
        )

        service = ExportVideoService()
        full_path = service.export_video(manifest, images)

        # Mezclar audio si está disponible
        if audio is not None:
            try:
                audio_np, sampling_rate = audio
                audio_path = os.path.splitext(full_path)[0] + ".wav"
                save_audio_wav(audio_path, audio_np, sampling_rate)
                
                muxed_path = os.path.splitext(full_path)[0] + "_muxed.mp4"
                mux_video_audio_with_ffmpeg(full_path, audio_path, muxed_path)
                os.replace(muxed_path, full_path)
                try:
                    os.unlink(audio_path)
                except Exception:
                    pass
                logger.info(f"[HEX-LTX] Audio mezclado en: {full_path}")
            except Exception as e:
                logger.warning(f"[HEX-LTX] No se pudo mezclar audio: {e}")

        # Construir payload UI para previsualización en ComfyUI
        if folder_paths is not None:
            base_output = folder_paths.get_output_directory()
            rel_path = os.path.relpath(full_path, base_output)
            subfolder = os.path.dirname(rel_path).replace("\\", "/")
            filename = os.path.basename(full_path)
        else:
            subfolder = ""
            filename = os.path.basename(full_path)

        logger.info(f"[HEX-LTX] Video guardado: {full_path}")

        return {
            "ui": {
                "gifs": [
                    {
                        "filename": filename,
                        "subfolder": subfolder,
                        "type": "output",
                    }
                ]
            }
        }
