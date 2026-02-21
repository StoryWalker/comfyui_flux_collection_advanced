"""
Flux VRAM Extreme Loader (BETA)
Version: 0.2.2-beta
Date: 2026-02-19
Author: TEAM_PRO (via Gemini CLI)
Description: Ultra-robust parameter validation. Prevents INT conversion crashes from old workflows.
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
    BETA: Extreme VRAM Optimization Loader specialized for Flux.
    v0.2.2: Extreme robustness against ComfyUI positional shifts.
    """

    FUNCTION = "load_models_vram"
    CATEGORY = "flux_collection_advanced/beta"
    DESCRIPTION = "BETA: Flux Loader with Architecture Fingerprinting and T5 optimizations."
    RETURN_TYPES = ("MODEL", "CLIP", "VAE",)
    OUTPUT_NODE = False

    # --- Static Data for TAESD ---
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
        unet_list = sorted(list(set(folder_paths.get_filename_list("unet_gguf") + folder_paths.get_filename_list("diffusion_models"))))
        clip_list = sorted(list(set(folder_paths.get_filename_list("clip_gguf") + folder_paths.get_filename_list("text_encoders"))))
        
        vaes = folder_paths.get_filename_list("vae")
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        available_taesd = []
        for variant, prefixes in cls._TAESD_VARIANTS_PREFIXES.items():
            if any(v.startswith(prefixes[0]) for v in approx_vaes) and any(v.startswith(prefixes[1]) for v in approx_vaes):
                available_taesd.append(variant)
        vae_list = ["None"] + vaes + available_taesd

        return {
            "required": {
                "unet_name": (unet_list, {"tooltip": "Select Flux UNET (.gguf or .safetensors)"}),
                "dequant_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                "clip_name1": (clip_list, {"tooltip": "Primary Encoder (CLIP-L)"}),
                "clip_name2": (["None"] + clip_list, {"tooltip": "Secondary Encoder (T5). Use 'None' for Flux.2."}),
                "t5_optimization": (["None", "Aggressive Offload", "Layer Truncation"], {"default": "Layer Truncation"}),
                "vae_name": (vae_list, {"tooltip": "Select VAE or TAESD variant."}),
                "auto_optimize": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # We move problematic widgets to optional to bypass strict INT validation if None/String leaked
                "t5_layers": ("INT", {"default": 16, "min": 1, "max": 24, "step": 1}),
                "patch_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                "patch_on_device": ("BOOLEAN", {"default": False}),
            }
        }

    def _detect_architecture(self, path: str) -> Dict[str, Any]:
        info = {"arch": "unknown", "is_flux2": False}
        try:
            if not path or not path.lower().endswith(".gguf"): return info
            from gguf import GGUFReader
            reader = GGUFReader(path)
            double = len([t for tensor in reader.tensors if (t := tensor.name).startswith("double_blocks.") and t.endswith(".img_mlp.2.weight")])
            single = len([t for tensor in reader.tensors if (t := tensor.name).startswith("single_blocks.") and t.endswith(".linear2.weight")])
            if double == 8 and single == 48:
                info["arch"] = "flux.2_optimized"
                info["is_flux2"] = True
            elif double == 19: info["arch"] = "flux.1_standard"
            return info
        except: return info

    def load_models_vram(self, unet_name, dequant_dtype, clip_name1, clip_name2, t5_optimization, vae_name, auto_optimize, t5_layers=16, patch_dtype="default", patch_on_device=False):
        # Robust Parameter Cleaning
        try:
            # If t5_layers was sent as a string (shift), try to recover or default
            if isinstance(t5_layers, str): t5_layers = 16
            t5_layers = int(t5_layers) if t5_layers is not None else 16
        except: t5_layers = 16

        # Safety for patch_dtype
        if patch_dtype not in ["default", "target", "float32", "float16", "bfloat16"]: patch_dtype = "default"

        # 1. Resolve Path and Detect Arch
        unet_path = folder_paths.get_full_path("unet_gguf", unet_name) or folder_paths.get_full_path("diffusion_models", unet_name)
        arch_info = self._detect_architecture(unet_path) if auto_optimize else {}

        # 2. Load UNET
        from nodes import NODE_CLASS_MAPPINGS
        if str(unet_name).lower().endswith(".gguf"):
            gguf_node = NODE_CLASS_MAPPINGS.get("UnetLoaderGGUFAdvanced")
            model = gguf_node().load_unet(unet_name, dequant_dtype, patch_dtype, patch_on_device)[0]
        else:
            model = comfy.sd.load_diffusion_model(unet_path)

        try:
            sampling = model.model.model_sampling
            if not hasattr(sampling, "shift"): sampling.set_parameters(shift=1.15)
        except: pass

        # 3. Load CLIP(s)
        use_single = (str(clip_name2) == "None" or not clip_name2)
        if use_single:
            if str(clip_name1).lower().endswith(".gguf"):
                clip = NODE_CLASS_MAPPINGS.get("CLIPLoaderGGUF")().load_clip(clip_name1, "flux")[0]
            else:
                clip_path = folder_paths.get_full_path_or_raise("clip", clip_name1)
                clip = comfy.sd.load_clip(ckpt_paths=[clip_path], clip_type=comfy.sd.CLIPType.FLUX)
        else:
            is_gguf = str(clip_name1).lower().endswith(".gguf") or str(clip_name2).lower().endswith(".gguf")
            if is_gguf:
                clip = NODE_CLASS_MAPPINGS.get("DualCLIPLoaderGGUF")().load_clip(clip_name1, clip_name2, "flux")[0]
            else:
                p1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
                p2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)
                clip = comfy.sd.load_clip(ckpt_paths=[p1, p2], clip_type=comfy.sd.CLIPType.FLUX)

        # 4. Apply T5 Optimizations
        if not use_single and str(t5_optimization) == "Layer Truncation":
            try:
                t5 = None
                if hasattr(clip.cond_stage_model, "t5xxl"): t5 = clip.cond_stage_model.t5xxl.transformer
                elif hasattr(clip.cond_stage_model, "transformer"): t5 = clip.cond_stage_model.transformer
                if t5 and hasattr(t5, "encoder") and t5_layers < 24:
                    t5.encoder.block = t5.encoder.block[:t5_layers]
            except: pass

        # 5. Load VAE
        vae = self._load_vae_robust(vae_name)
        return (model, clip, vae)

    def _load_vae_robust(self, name: str) -> Optional[Any]:
        if str(name) == "None": return None
        try:
            if name in self._TAESD_VARIANTS_PREFIXES:
                sd = {}
                approx_vaes = folder_paths.get_filename_list("vae_approx")
                prefixes = self._TAESD_VARIANTS_PREFIXES[name]
                e_name = next(filter(lambda a: a.startswith(prefixes[0]), approx_vaes))
                d_name = next(filter(lambda a: a.startswith(prefixes[1]), approx_vaes))
                enc = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", e_name))
                for k, v in enc.items(): sd[f"taesd_encoder.{k}"] = v
                dec = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", d_name))
                for k, v in dec_sd.items(): sd[f"taesd_decoder.{k}"] = v
                scale = self._TAESD_SCALING[name]
                sd["vae_scale"], sd["vae_shift"] = torch.tensor(scale["scale"]), torch.tensor(scale["shift"])
                return comfy.sd.VAE(sd=sd)
            else:
                return comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", name)))
        except: return None
