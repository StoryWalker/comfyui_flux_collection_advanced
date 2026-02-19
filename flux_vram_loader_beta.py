"""
Flux VRAM Extreme Loader (BETA)
Version: 0.1.0-beta
Date: 2026-02-19
Author: TEAM_PRO (via Gemini CLI)
Description: Specialized Flux 2 / GGUF Loader with Layer Truncation and Audit Logic.
"""
import logging
import torch
import comfy.sd
import comfy.utils
import comfy.model_management
import folder_paths
import nodes
from typing import Any, Dict, List, Tuple, Optional, Type

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FluxModelsLoader_VRAM_Beta(nodes.ComfyNodeABC):
    """
    BETA: Extreme VRAM Optimization Loader specialized for Flux 2 and GGUF.
    Combines GGUF quantized loading with T5 layer truncation.
    """

    FUNCTION = "load_models_vram"
    CATEGORY = "flux_collection_advanced/beta"
    DESCRIPTION = "Advanced Flux 2 / GGUF Loader with extreme VRAM optimizations."
    RETURN_TYPES = ("MODEL", "CLIP", "VAE",)
    OUTPUT_NODE = False

    # --- Static Data for TAESD Handling ---
    _TAESD_VARIANTS_PREFIXES = {
        "taesd": ["taesd_encoder.", "taesd_decoder."],
        "taesdxl": ["taesdxl_encoder.", "taesdxl_decoder."],
        "taesd3": ["taesd3_encoder.", "taesd3_decoder."],
        "taef1": ["taef1_encoder.", "taef1_decoder."],
    }
    _TAESD_SCALING = {
        "taesd": {"scale": 0.18215, "shift": 0.0},
        "taesdxl": {"scale": 0.13025, "shift": 0.0},
        "taesd3": {"scale": 1.5305, "shift": 0.0609},
        "taef1": {"scale": 0.3611, "shift": 0.1159},
    }

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """ Defines input types mirroring the structure of FluxModelsLoader but with GGUF/VRAM extras. """
        try:
            # GGUF and standard lists
            unet_list = sorted(list(set(folder_paths.get_filename_list("unet_gguf") + folder_paths.get_filename_list("diffusion_models"))))
            clip_list = sorted(list(set(folder_paths.get_filename_list("clip_gguf") + folder_paths.get_filename_list("text_encoders"))))
            
            # Use self-contained VAE list logic
            vaes = folder_paths.get_filename_list("vae")
            approx_vaes = folder_paths.get_filename_list("vae_approx")
            available_taesd = []
            for variant, prefixes in cls._TAESD_VARIANTS_PREFIXES.items():
                has_encoder = any(v.startswith(prefixes[0]) for v in approx_vaes)
                has_decoder = any(v.startswith(prefixes[1]) for v in approx_vaes)
                if has_encoder and has_decoder:
                    available_taesd.append(variant)
            vae_list = ["None"] + vaes + available_taesd
        except Exception as e:
            logger.exception("Failed to get model lists for Beta Loader.")
            unet_list, clip_list, vae_list = [], [], ["None"]

        return {
            "required": {
                "unet_name": (unet_list, {"tooltip": "Select Flux UNET (.gguf or .safetensors)"}),
                "dequant_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                "clip_name1": (clip_list, {"tooltip": "Primary Encoder (CLIP-L or Combined)"}),
                "clip_name2": (["None"] + clip_list, {"tooltip": "Secondary Encoder (T5-XXL). Set to 'None' for Flux 2 memory saving."}),
                "t5_optimization": (["None", "Aggressive Offload", "Layer Truncation"], {"default": "Layer Truncation"}),
                "vae_name": (vae_list, {"tooltip": "Select VAE or TAESD variant."}),
                "t5_layers": ("INT", {"default": 16, "min": 1, "max": 24, "step": 1, "tooltip": "Layers of T5 to use (24 is full). 16 is a good balance for VRAM."}),
            },
            "optional": {
                "patch_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                "patch_on_device": ("BOOLEAN", {"default": False}),
            }
        }

    def load_models_vram(self, unet_name, dequant_dtype, clip_name1, clip_name2, t5_optimization, vae_name, t5_layers, patch_dtype="default", patch_on_device=False):
        # Emergency Audit for positional shifts from old workflows
        # If t5_layers received a string (like a VAE name), we attempt to fix it
        if isinstance(t5_layers, str) and not t5_layers.isdigit():
             logger.warning(f"BETA: Detected positional shift. t5_layers received '{t5_layers}'. Reverting to default 16.")
             t5_layers = 16
        
        try: t5_layers = int(t5_layers)
        except: t5_layers = 16

        logger.info(f"Executing Beta Flux 2 GGUF Loader: {unet_name} | Layers: {t5_layers}")
        from nodes import NODE_CLASS_MAPPINGS

        # 1. Load UNET (GGUF Support)
        if unet_name.lower().endswith(".gguf"):
            gguf_node = NODE_CLASS_MAPPINGS.get("UnetLoaderGGUFAdvanced")
            if gguf_node is None: raise RuntimeError("Required plugin 'ComfyUI-GGUF' not found.")
            model = gguf_node().load_unet(unet_name, dequant_dtype, patch_dtype, patch_on_device)[0]
        else:
            unet_path = folder_paths.get_full_path("diffusion_models", unet_name) or folder_paths.get_full_path("unet", unet_name)
            model = comfy.sd.load_diffusion_model(unet_path)

        # Apply Sampling Shift (1.15)
        try:
            sampling = model.model.model_sampling
            if not hasattr(sampling, "shift"):
                sampling.set_parameters(shift=1.15)
                logger.info("Flux Sampling Shift 1.15 applied.")
        except: pass

        # 2. Load CLIP(s)
        use_single = (clip_name2 == "None" or not clip_name2)
        if use_single:
            if clip_name1.lower().endswith(".gguf"):
                clip = NODE_CLASS_MAPPINGS.get("CLIPLoaderGGUF")().load_clip(clip_name1, "flux")[0]
            else:
                clip_path = folder_paths.get_full_path_or_raise("clip", clip_name1)
                clip = comfy.sd.load_clip(ckpt_paths=[clip_path], clip_type=comfy.sd.CLIPType.FLUX)
        else:
            is_gguf = clip_name1.lower().endswith(".gguf") or clip_name2.lower().endswith(".gguf")
            if is_gguf:
                clip = NODE_CLASS_MAPPINGS.get("DualCLIPLoaderGGUF")().load_clip(clip_name1, clip_name2, "flux")[0]
            else:
                p1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
                p2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)
                clip = comfy.sd.load_clip(ckpt_paths=[p1, p2], clip_type=comfy.sd.CLIPType.FLUX)

        # 3. Apply T5 Optimizations
        if not use_single and t5_optimization == "Layer Truncation":
            try:
                t5 = None
                if hasattr(clip.cond_stage_model, "t5xxl"): t5 = clip.cond_stage_model.t5xxl.transformer
                elif hasattr(clip.cond_stage_model, "transformer"): t5 = clip.cond_stage_model.transformer
                if t5 and hasattr(t5, "encoder") and t5_layers < 24:
                    t5.encoder.block = t5.encoder.block[:t5_layers]
                    logger.info(f"VRAM Optimization: T5 truncated to {t5_layers} layers.")
            except Exception as e:
                logger.warning(f"T5 truncation failed: {e}")

        # 4. Load VAE
        if vae_name == "None":
            vae = None
        elif vae_name in self._TAESD_VARIANTS_PREFIXES:
            # Self-contained TAESD logic
            sd = {}
            approx_vaes = folder_paths.get_filename_list("vae_approx")
            prefixes = self._TAESD_VARIANTS_PREFIXES[vae_name]
            e_name = next(filter(lambda a: a.startswith(prefixes[0]), approx_vaes))
            d_name = next(filter(lambda a: a.startswith(prefixes[1]), approx_vaes))
            enc = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", e_name))
            for k, v in enc.items(): sd[f"taesd_encoder.{k}"] = v
            dec = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", d_name))
            for k, v in dec_sd.items(): sd[f"taesd_decoder.{k}"] = v
            scale_info = self._TAESD_SCALING[vae_name]
            sd["vae_scale"] = torch.tensor(scale_info["scale"])
            sd["vae_shift"] = torch.tensor(scale_info["shift"])
            vae = comfy.sd.VAE(sd=sd)
        else:
            vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
            vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(vae_path))

        return (model, clip, vae)
