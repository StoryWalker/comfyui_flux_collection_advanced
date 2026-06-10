# -*- coding: utf-8 -*-
"""
[HEX] LTX Native Video Saver.
Guarda video + audio generados por el pipeline nativo de ComfyUI LTX 2.3.
El audio viene en formato nativo: dict {"waveform": torch.Tensor, "sample_rate": int}.
"""
import logging
import os
import subprocess

import numpy as np
import torch

try:
    import folder_paths
except ImportError:
    folder_paths = None

try:
    from infrastructure.error_adapter import hex_error_handler
    from ltx_backend import save_audio_wav, mux_video_audio_with_ffmpeg
except ImportError:
    from .infrastructure.error_adapter import hex_error_handler
    from .ltx_backend import save_audio_wav, mux_video_audio_with_ffmpeg

logger = logging.getLogger(__name__)


class LTXNativeSaverHex:
    """
    [HEX] LTX Native Video Saver.
    Guarda frames de video en MP4 y mezcla con audio del pipeline nativo.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "section_export": ("STRING", {"default": "EXPORT CONFIGURATION"}),
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "LTX_Native"}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
            },
            "optional": {
                "audio": ("AUDIO",),
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

        filename_prefix = kwargs.get("filename_prefix", "LTX_Native")
        fps = kwargs.get("fps", 24)
        audio = kwargs.get("audio", None)

        # ------------------------------------------------------------------
        # 1. Resolver ruta de salida
        # ------------------------------------------------------------------
        if folder_paths is not None:
            output_dir = folder_paths.get_output_directory()
        else:
            output_dir = os.path.join(os.getcwd(), "output")
            os.makedirs(output_dir, exist_ok=True)

        # Evitar colisiones
        base_path = os.path.join(output_dir, f"{filename_prefix}_")
        counter = 1
        full_path = f"{base_path}{counter:05d}.mp4"
        while os.path.exists(full_path):
            counter += 1
            full_path = f"{base_path}{counter:05d}.mp4"

        # ------------------------------------------------------------------
        # 2. Guardar frames como video MP4 (sin audio)
        # ------------------------------------------------------------------
        try:
            import imageio
        except ImportError:
            imageio = None

        frames_np = (images.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        if imageio is not None:
            imageio.mimwrite(full_path, frames_np, fps=fps, quality=8, codec="libx264")
        else:
            # Fallback usando ffmpeg directo si imageio no esta disponible
            temp_frames_dir = os.path.join(output_dir, f"_temp_frames_{counter}")
            os.makedirs(temp_frames_dir, exist_ok=True)
            for i, frame in enumerate(frames_np):
                from PIL import Image as PILImage
                PILImage.fromarray(frame).save(os.path.join(temp_frames_dir, f"frame_{i:05d}.png"))
            cmd = [
                "ffmpeg", "-y", "-framerate", str(fps),
                "-i", os.path.join(temp_frames_dir, "frame_%05d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                full_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            import shutil
            shutil.rmtree(temp_frames_dir, ignore_errors=True)

        logger.info(f"[HEX-LTX-Native] Video guardado: {full_path}")

        # ------------------------------------------------------------------
        # 3. Mezclar audio si esta disponible
        # ------------------------------------------------------------------
        if audio is not None:
            try:
                waveform = audio.get("waveform")
                sample_rate = audio.get("sample_rate", 24000)

                if waveform is not None and sample_rate > 0:
                    # Convertir a numpy int16 [channels, samples]
                    if isinstance(waveform, torch.Tensor):
                        waveform = waveform.detach().cpu()
                        if waveform.dim() == 3:
                            waveform = waveform.squeeze(0)  # [channels, samples]
                        audio_np = waveform.numpy()
                    else:
                        audio_np = waveform

                    # Normalizar a int16
                    if audio_np.dtype in (np.float32, np.float64):
                        peak = np.abs(audio_np).max()
                        if peak > 1.0:
                            audio_np = audio_np / peak
                        audio_np = (audio_np * 32767.0).clip(-32768, 32767).astype(np.int16)

                    audio_path = os.path.splitext(full_path)[0] + ".wav"
                    save_audio_wav(audio_path, audio_np, sample_rate)

                    muxed_path = os.path.splitext(full_path)[0] + "_muxed.mp4"
                    mux_video_audio_with_ffmpeg(full_path, audio_path, muxed_path)
                    os.replace(muxed_path, full_path)
                    try:
                        os.unlink(audio_path)
                    except Exception:
                        pass
                    logger.info(f"[HEX-LTX-Native] Audio mezclado en: {full_path}")
            except Exception as e:
                logger.warning(f"[HEX-LTX-Native] No se pudo mezclar audio: {e}")

        # ------------------------------------------------------------------
        # 4. Construir payload UI
        # ------------------------------------------------------------------
        if folder_paths is not None:
            base_output = folder_paths.get_output_directory()
            rel_path = os.path.relpath(full_path, base_output)
            subfolder = os.path.dirname(rel_path).replace("\\", "/")
            filename = os.path.basename(full_path)
        else:
            subfolder = ""
            filename = os.path.basename(full_path)

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
