# -*- coding: utf-8 -*-
import logging
import torch
import comfy.sd
import comfy.utils
import comfy.clip_vision
import folder_paths
import os
from nodes import NODE_CLASS_MAPPINGS
from typing import Any, Dict, Tuple, Optional

logger = logging.getLogger(__name__)

class ComfyModelAdapter:
    """
    Infrastructure Adapter for Model Loading.
    Encapsulates ComfyUI's loading mechanisms to prevent recursion and conflicts.
    """
    
    @staticmethod
    def load_unet(name: str, weight_dtype: str = "default") -> Any:
        """ 
        Securely loads UNET models, detecting GGUF automatically.
        Ensures parity with native ComfyUI and GGUF loaders.
        """
        logger.info(f"[HEX] Infrastructure: Loading UNET '{name}' with dtype '{weight_dtype}'")
        
        # 1. Handle GGUF
        if name.lower().endswith(".gguf"):
            gguf_loader_cls = NODE_CLASS_MAPPINGS.get("UnetLoaderGGUF")
            if not gguf_loader_cls:
                raise RuntimeError("UnetLoaderGGUF node not found. Please install comfyui-gguf.")
            
            loader_instance = gguf_loader_cls()
            return loader_instance.load_unet(unet_name=name)[0]
        
        # 2. Handle Safetensors / Standard
        path = folder_paths.get_full_path("diffusion_models", name) or folder_paths.get_full_path("unet", name)
        if not path:
            raise FileNotFoundError(f"Model file not found: {name}")

        model_options = {}
        if weight_dtype == "fp8_e4m3fn": 
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "bf16": 
            model_options["dtype"] = torch.bfloat16
        
        return comfy.sd.load_diffusion_model(path, model_options=model_options)

    @staticmethod
    def load_clip_wan(name: str, optimization: str = "Layer Truncation", layers: int = 16) -> Any:
        """ 
        Loads CLIP specifically for WAN architecture.
        Detects GGUF and applies T5 optimizations with strict tracing.
        """
        # 1. Detect GGUF CLIP
        if name.lower().endswith(".gguf"):
            clip_gguf_loader = NODE_CLASS_MAPPINGS.get("CLIPLoaderGGUF")
            if not clip_gguf_loader:
                raise RuntimeError("CLIPLoaderGGUF node not found. Please install comfyui-gguf.")
            
            logger.info(f"[HEX] CLIP Trace: Loading GGUF '{name}' as WAN type")
            clip = clip_gguf_loader().load_clip(clip_name=name, type="wan")[0]
        else:
            path = folder_paths.get_full_path_or_raise("text_encoders", name)
            logger.info(f"[HEX] CLIP Trace: Path resolved -> {path}")
            
            # Diagnostic size check
            if os.path.exists(path):
                size_gb = os.path.getsize(path) / (1024**3)
                logger.info(f"[HEX] CLIP Trace: File size -> {size_gb:.2f} GB")
            
            # Task-Source: T#2
            # Force WAN type (UMT5)
            # We explicitly pass clip_type=WAN to ensure 4096 dimensions.
            try:
                clip_type = comfy.sd.CLIPType.WAN
            except AttributeError:
                # Versiones antiguas de ComfyUI no tienen CLIPType.WAN definido
                logger.warning("[HEX] CLIPType.WAN no disponible en esta version de ComfyUI, usando fallback string 'wan'")
                clip_type = "wan"
            
            logger.info(f"[HEX] CLIP Trace: Enforcing type '{clip_type}' for Wan 2.2")
            # Using the standard loader without manual embedding dir to avoid path errors
            clip = comfy.sd.load_clip(ckpt_paths=[path], clip_type=clip_type)
            
            # Verification: Force failure if it detects SD1
            if hasattr(clip, "cond_stage_model"):
                arch_name = type(clip.cond_stage_model).__name__
                logger.info(f"[HEX] CLIP Trace: Architecture verified as -> {arch_name}")
                if "SD1ClipModel" in arch_name:
                    logger.error("[HEX] CRITICAL: ComfyUI identified Wan model as SD1. Check file integrity.")
                    raise RuntimeError("Architecture Mismatch: SD1 detected for Wan model.")

        # 2. Apply T5 Optimizations
        if optimization == "Layer Truncation":
            logger.info(f"[HEX] CLIP Trace: Truncating to {layers} layers.")
            clip = clip.clone()
            try:
                clip.cond_stage_model.t5_layers = layers
            except AttributeError:
                logger.warning("[HEX] CLIP Trace: T5 Truncation failed (attribute not found)")
            
        return clip

    @staticmethod
    def apply_lora_robust(model: Any, clip: Any, lora_name: str, strength: float) -> Tuple[Any, Any]:
        """ Applies LoRA patching with key cleanup to avoid 'not loaded' errors """
        if lora_name == "None": return model, clip
        
        path = folder_paths.get_full_path("loras", lora_name)
        lora_sd = comfy.utils.load_torch_file(path, safe_load=True)
        # Cleanup prefixes
        clean_sd = {k.replace("diffusion_model.", ""): v for k, v in lora_sd.items()}
        
        return comfy.sd.load_lora_for_models(model, clip, clean_sd, strength, strength)

    @staticmethod
    def apply_sampling_shift(model: Any, shift: float) -> Any:
        """ Patches model sampling shift safely in-place """
        patched = model.clone()
        try:
            patched.model.model_sampling.set_parameters(shift=shift)
        except Exception as e:
            logger.warning(f"Fallback shift patching used: {e}")
        return patched

    @staticmethod
    def load_vae_and_vision(vae_name: str, cv_name: str) -> Tuple[Any, Any]:
        """ Loads standard VAE and CLIP Vision models with safety checks """
        vae, cv = None, None
        
        if vae_name:
            vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
            vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))
        
        if cv_name:
            cv_path = folder_paths.get_full_path_or_raise("clip_vision", cv_name)
            cv = comfy.clip_vision.load(cv_path)
            
        return vae, cv
