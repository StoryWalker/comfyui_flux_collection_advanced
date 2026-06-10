# -*- coding: utf-8 -*-
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
from ltx_pipelines.distilled import DistilledPipeline
from ltx_core.quantization import QuantizationPolicy

# Centralized NexusForge models directory
NEXUS_MODELS_DIR = "C:/Users/GATEWAY/AppData/Local/NexusForge/models"

def scan_ltx_files(comfyui_folder_type, extension_or_dir, default_file=None):
    """
    Scans both NexusForge AppData and ComfyUI standard directories to return
    a list of available files/folders for dropdown menus in the UI.
    """
    files = []
    
    # 1. Scan NexusForge models folder
    if os.path.exists(NEXUS_MODELS_DIR):
        for name in os.listdir(NEXUS_MODELS_DIR):
            path = os.path.join(NEXUS_MODELS_DIR, name)
            if extension_or_dir == "dir":
                if os.path.isdir(path):
                    files.append(name)
            else:
                if os.path.isfile(path) and name.endswith(extension_or_dir):
                    files.append(name)
                    
    # 2. Scan ComfyUI folders
    if comfyui_folder_type:
        try:
            comfy_files = folder_paths.get_filename_list(comfyui_folder_type)
            for f in comfy_files:
                if extension_or_dir == "dir":
                    full_p = folder_paths.get_full_path(comfyui_folder_type, f)
                    if full_p and os.path.isdir(full_p):
                        files.append(f)
                else:
                    if f.endswith(extension_or_dir) and f not in files:
                        files.append(f)
        except Exception:
            pass
            
    # Include default file if not already detected
    if default_file and default_file not in files:
        files.insert(0, default_file)
        
    if not files:
        files.append("None")
        
    return sorted(list(set(files)))


def resolve_ltx_path(filename, comfyui_folder_type=None, default_file=None):
    """
    Resolves the actual path on disk for the chosen model name.
    """
    if not filename or filename == "None":
        return None
        
    # If absolute path is directly written/passed
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
        
    # Check in NexusForge models directory
    nexus_path = os.path.join(NEXUS_MODELS_DIR, filename)
    if os.path.exists(nexus_path):
        return nexus_path
        
    # Check in ComfyUI folders
    if comfyui_folder_type:
        try:
            full_path = folder_paths.get_full_path(comfyui_folder_type, filename)
            if full_path and os.path.exists(full_path):
                return full_path
        except Exception:
            pass
            
    # Fallback to default path
    return nexus_path


class LTXDistilledGGUFPipelineLoader:
    """
    [LTX] Distilled GGUF Pipeline Loader:
    Carga el transformador cuantizado (Q4_K_M GGUF) y sus modelos acompañantes (VAEs, Conector, Gemma)
    en una instancia optimizada de DistilledPipeline.
    """
    @classmethod
    def INPUT_TYPES(s):
        gguf_list = scan_ltx_files("checkpoints", ".gguf", "ltx-2.3-22b-distilled-Q4_K_M.gguf")
        gemma_list = scan_ltx_files("clip", "dir", "gemma-3-12b-it-qat-q4_0-unquantized")
        upscaler_list = scan_ltx_files("upscale_models", ".safetensors", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors")
        vae_list = scan_ltx_files("vae", ".safetensors", "ltx-2.3-22b-dev_video_vae.safetensors")
        audio_vae_list = scan_ltx_files("vae", ".safetensors", "ltx-2.3-22b-dev_audio_vae.safetensors")
        connector_list = scan_ltx_files("clip", ".safetensors", "ltx-2.3-22b-dev_embeddings_connectors.safetensors")

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

    def load_pipeline(self, gguf_checkpoint, gemma_directory, spatial_upscaler, vae_video, audio_vae, connector, device):
        # Resolver las rutas completas
        gguf_checkpoint_path = resolve_ltx_path(gguf_checkpoint, "checkpoints")
        gemma_path_or_dir = resolve_ltx_path(gemma_directory, "clip")
        spatial_upscaler_path = resolve_ltx_path(spatial_upscaler, "upscale_models")
        vae_video_path = resolve_ltx_path(vae_video, "vae")
        audio_vae_path = resolve_ltx_path(audio_vae, "vae")
        connector_path = resolve_ltx_path(connector, "clip")

        # Añadimos el directorio G: del proyecto al PATH para importar los módulos auxiliares de GGUF
        sys.path.append(r"G:\Developing\Claude\ltx_video\NexusForge\backend")
        from services.fast_video_pipeline.ltx_distilled_gguf_video_pipeline import LTXDistilledGGUFVideoPipeline

        torch_device = torch.device(device)
        pipeline = LTXDistilledGGUFVideoPipeline.create(
            checkpoint_path=gguf_checkpoint_path,
            gemma_root=gemma_path_or_dir,
            upsampler_path=spatial_upscaler_path,
            device=torch_device,
            vae_video_path=vae_video_path,
            audio_vae_path=audio_vae_path,
            connector_path=connector_path,
        )
        return (pipeline,)


class LTXFastPipelineLoader:
    """
    [LTX] Fast Pipeline Loader:
    Carga el transformador estándar de LTX (SafeTensors) en fp8/bf16 de forma nativa.
    """
    @classmethod
    def INPUT_TYPES(s):
        checkpoint_list = scan_ltx_files("checkpoints", ".safetensors", "ltx-2.3-22b-distilled.safetensors")
        gemma_list = scan_ltx_files("clip", "dir", "gemma-3-12b-it-qat-q4_0-unquantized")
        upscaler_list = scan_ltx_files("upscale_models", ".safetensors", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors")

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

    def load_pipeline(self, checkpoint, gemma_directory, spatial_upscaler, device):
        checkpoint_path = resolve_ltx_path(checkpoint, "checkpoints")
        gemma_path_or_dir = resolve_ltx_path(gemma_directory, "clip")
        spatial_upscaler_path = resolve_ltx_path(spatial_upscaler, "upscale_models")

        sys.path.append(r"G:\Developing\Claude\ltx_video\NexusForge\backend")
        from services.fast_video_pipeline.ltx_fast_video_pipeline import LTXFastVideoPipeline

        torch_device = torch.device(device)
        pipeline = LTXFastVideoPipeline.create(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_path_or_dir,
            upsampler_path=spatial_upscaler_path,
            device=torch_device,
        )
        return (pipeline,)


class LTXVideoSampler:
    """
    [LTX] Video Sampler:
    Toma un pipeline cargado y genera los frames del video en ComfyUI,
    soportando de manera opcional condicionamiento de imagen (Image-to-Video).
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ltx_pipeline": ("LTX_PIPELINE",),
                "prompt": ("STRING", {"multiline": True, "default": "a slow camera pan across a realistic forest portrait, high definition"}),
                "width": ("INT", {"default": 768, "min": 64, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 32}),
                "num_frames": ("INT", {"default": 97, "min": 9, "max": 257, "step": 8}), # 8k+1
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "sample"
    CATEGORY = "flux_collection_advanced/ltx"

    def sample(self, ltx_pipeline, prompt, width, height, num_frames, frame_rate, seed, image=None, strength=1.0):
        # Creamos la configuración de tiling por defecto optimizada para Blackwell/OOM
        sys.path.append(r"G:\Developing\Claude\ltx_video\NexusForge\backend")
        from services.ltx_pipeline_common import default_tiling_config
        from api_types import ImageConditioningInput
        
        tiling_config = default_tiling_config()
        
        # Preparación de imágenes condicionales (I2V) si se proporciona una imagen
        images_input = []
        temp_file = None
        if image is not None:
            # ComfyUI image shape is [B, H, W, C]. We extract the first frame.
            frame_tensor = image[0]
            # Convert to [0, 255] uint8 numpy array
            np_frame = (frame_tensor.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            pil_image = Image.fromarray(np_frame)
            
            # Guardamos a un archivo temporal
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, f"ltx_cond_{seed}.png")
            # Redimensionamos la imagen condicional para que encaje con las dimensiones de salida de LTX
            pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
            pil_image.save(temp_file)
            
            images_input = [ImageConditioningInput(path=temp_file, frame_idx=0, strength=strength)]

        try:
            # Ejecutamos la inferencia
            # ltx_pipeline es una instancia de LTXFastVideoPipeline o LTXDistilledGGUFVideoPipeline
            video, audio = ltx_pipeline._run_inference(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                images=images_input,
                tiling_config=tiling_config,
            )
            
            # Recolectamos todos los frames del tensor
            if isinstance(video, torch.Tensor):
                video_tensor = video
            else:
                # Es un Iterator[torch.Tensor]
                video_tensor = torch.cat(list(video), dim=0)
            
            # video_tensor tiene forma [frames, height, width, 3] en escala [0, 255]
            # ComfyUI espera [frames, height, width, 3] float32 [0.0, 1.0]
            comfyui_images = video_tensor.to(dtype=torch.float32) / 255.0
            
            return (comfyui_images,)
            
        finally:
            # Nos aseguramos de eliminar el archivo temporal
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass
