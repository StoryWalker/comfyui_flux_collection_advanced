# -*- coding: utf-8 -*-
"""
[LTX] Legacy Nodes — Thin Hexagonal Wrappers.

Estos nodos mantienen la interfaz legacy estable (nombres, INPUT_TYPES, RETURN_TYPES)
para no romper workflows existentes, pero internamente delegan 100% a la arquitectura
hexagonal (Domain → Application → Infrastructure).

Nodos HEX puros: hex_ltx_loader.py, hex_ltx_sampler.py, hex_ltx_video_saver.py
Nodos Legacy wrappers: este archivo.
"""
import os
import sys
import torch
import tempfile
import numpy as np
from PIL import Image

# CRITICAL IMPORT ORDER: Import fuse_loras first to break circular imports in ltx_core
try:
    import ltx_core.loader.fuse_loras
except ImportError:
    pass

import folder_paths

# ---------------------------------------------------------------------------
# Registrar .gguf como extensión soportada en las categorías relevantes.
# ComfyUI por defecto solo soporta .safetensors, .ckpt, .pt, etc.
# Sin esto, get_filename_list() nunca retornará archivos .gguf.
# ---------------------------------------------------------------------------
_GGUF_EXT = {".gguf"}
for _cat in ("diffusion_models", "checkpoints", "unet"):
    try:
        _info = folder_paths.folder_names_and_paths.get(_cat)
        if _info is not None:
            _paths, _exts = _info
            if ".gguf" not in _exts:
                _exts.update(_GGUF_EXT)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Hexagonal Imports (Domain → Application → Infrastructure)
# ---------------------------------------------------------------------------
try:
    from domain.models import LTXPipelineConfig, LTXGenerationSettings, VideoExportManifest
    from application.ltx_video_generation_service import LTXVideoGenerationService
    from application.export_service import ExportVideoService
    from infrastructure.ltx_pipeline_adapter import LTXPipelineAdapter
    from infrastructure.error_adapter import hex_error_handler
    from ltx_backend import save_audio_wav, mux_video_audio_with_ffmpeg
except ImportError:
    from .domain.models import LTXPipelineConfig, LTXGenerationSettings, VideoExportManifest
    from .application.ltx_video_generation_service import LTXVideoGenerationService
    from .application.export_service import ExportVideoService
    from .infrastructure.ltx_pipeline_adapter import LTXPipelineAdapter
    from .infrastructure.error_adapter import hex_error_handler
    from .ltx_backend import save_audio_wav, mux_video_audio_with_ffmpeg


def scan_ltx_files(folder_types, extension_or_dir, default_file=None):
    """
    Escanea carpetas nativas de ComfyUI para retornar una lista de modelos disponibles.
    Acepta una categoría única (str) o múltiples categorías (list/tuple) para buscar
    en varios sitios simultáneamente (ej: ['diffusion_models', 'checkpoints']).
    """
    if isinstance(folder_types, str):
        folder_types = [folder_types]

    files = []

    for folder_type in folder_types:
        try:
            comfy_files = folder_paths.get_filename_list(folder_type)
            for f in comfy_files:
                if extension_or_dir == "dir":
                    full_p = folder_paths.get_full_path(folder_type, f)
                    if full_p and os.path.isdir(full_p):
                        if f not in files:
                            files.append(f)
                else:
                    if f.endswith(extension_or_dir) and f not in files:
                        files.append(f)
        except Exception:
            pass

    # Solo insertar el default si no hay un archivo real cuyo basename coincida.
    if default_file and default_file not in files:
        basenames = {os.path.basename(f) for f in files}
        if os.path.basename(default_file) not in basenames:
            files.insert(0, default_file)

    if not files:
        files.append("None")

    return sorted(list(set(files)))


def resolve_ltx_path(filename, folder_types=None):
    """
    Resuelve la ruta completa de un modelo en ComfyUI.
    Acepta una categoría única (str) o múltiples categorías (list/tuple).
    Busca secuencialmente hasta encontrar el archivo en disco.
    Si el archivo no existe, retorna la ruta donde se esperaba encontrar.
    """
    if not filename or filename == "None":
        return None

    if os.path.isabs(filename):
        return filename

    if folder_types is None:
        return None

    if isinstance(folder_types, str):
        folder_types = [folder_types]

    # Fase 1: buscar el archivo real en todas las categorías
    for folder_type in folder_types:
        try:
            full_path = folder_paths.get_full_path(folder_type, filename)
            if full_path and os.path.exists(full_path):
                return full_path
        except Exception:
            pass

    # Fase 2: construir la ruta esperada (para mensajes de error descriptivos)
    for folder_type in folder_types:
        try:
            paths = folder_paths.get_folder_paths(folder_type)
            if paths:
                return os.path.join(paths[0], filename)
        except Exception:
            pass

    # Fallback descriptivo
    return os.path.join("models", folder_types[0] if folder_types else "", filename)


# =============================================================================
# [LTX] Distilled GGUF Pipeline Loader  (Legacy Wrapper → Hexagonal)
# =============================================================================
class LTXDistilledGGUFPipelineLoader:
    """
    [LTX] Distilled GGUF Pipeline Loader (Legacy):
    Carga el transformador cuantizado (Q4_K_M GGUF) y sus modelos acompañantes.
    Internamente delega a LTXPipelineAdapter (infraestructura hexagonal).
    """
    _GGUF_CATEGORIES = ["diffusion_models", "checkpoints", "unet"]
    _GEMMA_CATEGORIES = ["clip", "text_encoders"]
    _UPSCALER_CATEGORIES = ["latent_upscale_models", "upscale_models"]
    _VAE_CATEGORIES = ["vae"]
    _CONNECTOR_CATEGORIES = ["text_encoders", "clip"]

    @classmethod
    def INPUT_TYPES(s):
        gguf_list = scan_ltx_files(s._GGUF_CATEGORIES, ".gguf", "ltx-2.3-22b-distilled-Q4_K_M.gguf")
        gemma_list = scan_ltx_files(s._GEMMA_CATEGORIES, "dir", "gemma-3-12b-it-qat-q4_0-unquantized")
        upscaler_list = scan_ltx_files(s._UPSCALER_CATEGORIES, ".safetensors", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors")
        vae_list = scan_ltx_files(s._VAE_CATEGORIES, ".safetensors", "ltx-2.3-22b-dev_video_vae.safetensors")
        audio_vae_list = scan_ltx_files(s._VAE_CATEGORIES, ".safetensors", "ltx-2.3-22b-dev_audio_vae.safetensors")
        connector_list = scan_ltx_files(s._CONNECTOR_CATEGORIES, ".safetensors", "ltx-2.3-22b-dev_embeddings_connectors.safetensors")

        return {
            "required": {
                "gguf_checkpoint": (gguf_list,),
                "gemma_directory": (gemma_list,),
                "spatial_upscaler": (upscaler_list,),
                "vae_video": (vae_list,),
                "audio_vae": (audio_vae_list,),
                "connector": (connector_list,),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("LTX_PIPELINE",)
    RETURN_NAMES = ("ltx_pipeline",)
    FUNCTION = "load_pipeline"
    CATEGORY = "flux_collection_advanced/ltx"

    @hex_error_handler
    def load_pipeline(self, gguf_checkpoint, gemma_directory, spatial_upscaler, vae_video, audio_vae, connector, device):
        gguf_checkpoint_path = resolve_ltx_path(gguf_checkpoint, self._GGUF_CATEGORIES)
        gemma_path_or_dir = resolve_ltx_path(gemma_directory, self._GEMMA_CATEGORIES)
        spatial_upscaler_path = resolve_ltx_path(spatial_upscaler, self._UPSCALER_CATEGORIES)
        vae_video_path = resolve_ltx_path(vae_video, self._VAE_CATEGORIES)
        audio_vae_path = resolve_ltx_path(audio_vae, self._VAE_CATEGORIES)
        connector_path = resolve_ltx_path(connector, self._CONNECTOR_CATEGORIES)

        # Validar existencia
        missing_resources = []
        for name, p in [
            ("gguf_checkpoint", gguf_checkpoint_path),
            ("gemma_directory", gemma_path_or_dir),
            ("spatial_upscaler", spatial_upscaler_path),
            ("vae_video", vae_video_path),
            ("audio_vae", audio_vae_path),
            ("connector", connector_path)
        ]:
            if not p or not os.path.exists(p):
                missing_resources.append((name, p))

        if missing_resources:
            msg_lines = ["[LTX Loader Error] Faltan los siguientes recursos:"]
            for name, p in missing_resources:
                msg_lines.append(f"  • {name}: {p}")
            msg_lines.append("")
            msg_lines.append("Consulta ltx_models.md para la lista completa de modelos requeridos.")
            msg_lines.append("Descarga los modelos LTX 2.3 y colócalos en las carpetas de ComfyUI indicadas.")
            raise FileNotFoundError("\n".join(msg_lines))

        config = LTXPipelineConfig(
            checkpoint_path=gguf_checkpoint_path,
            gemma_root=gemma_path_or_dir,
            upsampler_path=spatial_upscaler_path,
            vae_video_path=vae_video_path,
            audio_vae_path=audio_vae_path,
            connector_path=connector_path,
            pipeline_type="distilled_gguf",
            device=device,
        )

        adapter = LTXPipelineAdapter()
        pipeline = adapter.load_pipeline(config)
        return (pipeline,)


# =============================================================================
# [LTX] Fast Pipeline Loader  (Legacy Wrapper → Hexagonal)
# =============================================================================
class LTXFastPipelineLoader:
    """
    [LTX] Fast Pipeline Loader (Legacy):
    Carga el transformador estándar de LTX (SafeTensors) en fp8/bf16.
    Internamente delega a LTXPipelineAdapter (infraestructura hexagonal).
    """
    _CHECKPOINT_CATEGORIES = ["diffusion_models", "checkpoints", "unet"]
    _GEMMA_CATEGORIES = ["clip", "text_encoders"]
    _UPSCALER_CATEGORIES = ["latent_upscale_models", "upscale_models"]

    @classmethod
    def INPUT_TYPES(s):
        checkpoint_list = scan_ltx_files(s._CHECKPOINT_CATEGORIES, ".safetensors", "ltx-2.3-22b-distilled.safetensors")
        gemma_list = scan_ltx_files(s._GEMMA_CATEGORIES, "dir", "gemma-3-12b-it-qat-q4_0-unquantized")
        upscaler_list = scan_ltx_files(s._UPSCALER_CATEGORIES, ".safetensors", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors")

        return {
            "required": {
                "checkpoint": (checkpoint_list,),
                "gemma_directory": (gemma_list,),
                "spatial_upscaler": (upscaler_list,),
                "device": (["cuda", "cpu"], {"default": "cuda"}),
            }
        }

    RETURN_TYPES = ("LTX_PIPELINE",)
    RETURN_NAMES = ("ltx_pipeline",)
    FUNCTION = "load_pipeline"
    CATEGORY = "flux_collection_advanced/ltx"

    @hex_error_handler
    def load_pipeline(self, checkpoint, gemma_directory, spatial_upscaler, device):
        checkpoint_path = resolve_ltx_path(checkpoint, self._CHECKPOINT_CATEGORIES)
        gemma_path_or_dir = resolve_ltx_path(gemma_directory, self._GEMMA_CATEGORIES)
        spatial_upscaler_path = resolve_ltx_path(spatial_upscaler, self._UPSCALER_CATEGORIES)

        missing_resources = []
        for name, p in [
            ("checkpoint", checkpoint_path),
            ("gemma_directory", gemma_path_or_dir),
            ("spatial_upscaler", spatial_upscaler_path)
        ]:
            if not p or not os.path.exists(p):
                missing_resources.append((name, p))

        if missing_resources:
            msg_lines = ["[LTX Fast Loader Error] Faltan los siguientes recursos:"]
            for name, p in missing_resources:
                msg_lines.append(f"  • {name}: {p}")
            msg_lines.append("")
            msg_lines.append("Consulta ltx_models.md para la lista completa de modelos requeridos.")
            msg_lines.append("Descarga los modelos LTX 2.3 y colócalos en las carpetas de ComfyUI indicadas.")
            raise FileNotFoundError("\n".join(msg_lines))

        config = LTXPipelineConfig(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_path_or_dir,
            upsampler_path=spatial_upscaler_path,
            pipeline_type="fast",
            device=device,
        )

        adapter = LTXPipelineAdapter()
        pipeline = adapter.load_pipeline(config)
        return (pipeline,)


# =============================================================================
# [LTX] Video Sampler  (Legacy Wrapper → Hexagonal)
# =============================================================================
class LTXVideoSampler:
    """
    [LTX] Video Sampler (Legacy):
    Toma un pipeline cargado y genera los frames del video en ComfyUI.
    Internamente delega a LTXVideoGenerationService (aplicación hexagonal).
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ltx_pipeline": ("LTX_PIPELINE",),
                "prompt": ("STRING", {"multiline": True, "default": "a slow camera pan across a realistic forest portrait, high definition"}),
                "width": ("INT", {"default": 768, "min": 64, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 32}),
                "num_frames": ("INT", {"default": 97, "min": 9, "max": 257, "step": 8}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "LTX_AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "sample"
    CATEGORY = "flux_collection_advanced/ltx"

    @hex_error_handler
    def sample(self, ltx_pipeline, prompt, width, height, num_frames, frame_rate, seed, image=None, strength=1.0):
        # Preparar imagen condicional (I2V) si se proporciona
        image_path = ""
        temp_file = None
        if image is not None:
            frame_tensor = image[0]  # [H, W, C]
            np_frame = (frame_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            pil_image = Image.fromarray(np_frame)
            pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, f"ltx_cond_{seed}.png")
            pil_image.save(temp_file)
            image_path = temp_file

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
            )

            video_output, audio_data = service.generate(ltx_pipeline, settings)
            return (video_output, audio_data)

        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass


# =============================================================================
# [LTX] Video Saver  (Legacy Wrapper → Hexagonal)
# =============================================================================
class LTXVideoSaver:
    """
    [LTX] Video Saver (Legacy):
    Toma los fotogramas del video en formato IMAGE de ComfyUI, los codifica en MP4
    y permite mezclar audio. Internamente delega a ExportVideoService (aplicación hexagonal).
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
                "filename_prefix": ("STRING", {"default": "LTX_Video"}),
            },
            "optional": {
                "audio": ("LTX_AUDIO",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_video"
    OUTPUT_NODE = True
    CATEGORY = "flux_collection_advanced/ltx"

    @hex_error_handler
    def save_video(self, images, fps, filename_prefix, audio=None):
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
            except Exception as e:
                print(f"[LTX VideoSaver] Advertencia: no se pudo mezclar audio: {e}")

        # Construir payload UI para previsualización en ComfyUI
        base_output = folder_paths.get_output_directory()
        rel_path = os.path.relpath(full_path, base_output)
        subfolder = os.path.dirname(rel_path).replace("\\", "/")
        filename = os.path.basename(full_path)

        return {"ui": {"gifs": [{"filename": filename, "subfolder": subfolder, "type": "output"}]}}
