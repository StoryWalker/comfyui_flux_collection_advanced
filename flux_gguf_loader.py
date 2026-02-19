import logging
import os
import torch
import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_management
import nodes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FluxGGUFLoader(nodes.ComfyNodeABC):
    """
    Advanced All-in-One loader for FLUX.1 & FLUX.2.
    Fully aligned with official GGUF logic.
    Loads UNET (GGUF/Safetensors), CLIP(s), and VAE in a single node.
    Supports Single CLIP bypass for Flux 2.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        def get_files(keys):
            files = []
            for key in keys:
                try: files += folder_paths.get_filename_list(key)
                except: pass
            return sorted(list(set(files)))

        unet_list = get_files(["unet_gguf", "diffusion_models", "unet"])
        clip_list = get_files(["clip_gguf", "text_encoders", "clip"])
        vae_list = get_files(["vae", "vae_approx"])
        
        # Add TAESD variants
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        for variant in ["taesd", "taesdxl", "taesd3", "taef1"]:
            if any(v.startswith(variant + "_encoder.") for v in approx_vaes):
                vae_list.append(variant)

        return {
            "required": {
                "unet_name": (unet_list, {"tooltip": "Select Flux UNET (.gguf or .safetensors)"}),
                "clip_name1": (clip_list, {"tooltip": "Primary Encoder (CLIP-L or Combined)"}),
                "clip_name2": (["None"] + clip_list, {"tooltip": "Secondary Encoder (T5-XXL). Set to 'None' for Flux 2."}),
                "vae_name": (vae_list, {"tooltip": "Select VAE or TAESD variant."}),
                "clip_type": (["flux", "flux2", "sd3", "sdxl"], {"default": "flux"}),
                "base_type": (["flux", "flux2", "wan2.1"], {"default": "flux"}),
                "dequant_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                "patch_dtype": (["default", "target", "float32", "float16", "bfloat16"], {"default": "default"}),
                "patch_on_device": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    FUNCTION = "load_all_flux"
    CATEGORY = "flux_collection_advanced"
    DESCRIPTION = "Aligned All-in-One FLUX GGUF Loader. Supports single CLIP for Flux 2."

    def load_all_flux(self, unet_name, clip_name1, clip_name2, vae_name, clip_type, base_type, dequant_dtype, patch_dtype, patch_on_device):
        logger.info(f"Flux Advanced Loader: Preparing {unet_name} stack...")
        from nodes import NODE_CLASS_MAPPINGS

        # 1. Load UNET
        model = None
        if unet_name.lower().endswith(".gguf"):
            logger.info("Using GGUF Engine for UNET loading...")
            gguf_node = NODE_CLASS_MAPPINGS.get("UnetLoaderGGUFAdvanced")
            if gguf_node is None: raise RuntimeError("Plugin ComfyUI-GGUF not found.")
            model = gguf_node().load_unet(unet_name, dequant_dtype, patch_dtype, patch_on_device)[0]
        else:
            unet_path = folder_paths.get_full_path("diffusion_models", unet_name) or folder_paths.get_full_path("unet", unet_name)
            model = comfy.sd.load_diffusion_model(unet_path)

        if base_type in ["flux", "flux2"]:
            self._apply_flux_sampling(model)

        # 2. Load CLIP(s) - Intelligent Bypass
        use_single_clip = (clip_name2 == "None" or not clip_name2)
        c_type_str = "flux" if clip_type in ["flux", "flux2"] else clip_type
        
        if use_single_clip:
            logger.info(f"Single CLIP mode active for {clip_name1}")
            if clip_name1.lower().endswith(".gguf"):
                clip_node = NODE_CLASS_MAPPINGS.get("CLIPLoaderGGUF")
                clip = clip_node().load_clip(clip_name1, c_type_str)[0]
            else:
                clip_path = folder_paths.get_full_path_or_raise("clip", clip_name1)
                c_type = getattr(comfy.sd.CLIPType, c_type_str.upper(), comfy.sd.CLIPType.FLUX)
                clip = comfy.sd.load_clip(ckpt_paths=[clip_path], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=c_type)
        else:
            logger.info(f"Dual CLIP mode active: {clip_name1} + {clip_name2}")
            is_clip_gguf = clip_name1.lower().endswith(".gguf") or clip_name2.lower().endswith(".gguf")
            if is_clip_gguf:
                clip_node = NODE_CLASS_MAPPINGS.get("DualCLIPLoaderGGUF")
                clip = clip_node().load_clip(clip_name1, clip_name2, c_type_str)[0]
            else:
                clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
                clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)
                c_type = getattr(comfy.sd.CLIPType, c_type_str.upper(), comfy.sd.CLIPType.FLUX)
                clip = comfy.sd.load_clip(ckpt_paths=[clip_path1, clip_path2], embedding_directory=folder_paths.get_folder_paths("embeddings"), clip_type=c_type)

        # 3. Load VAE
        if vae_name in ["taesd", "taesdxl", "taesd3", "taef1"]:
            vae = self._load_taesd(vae_name)
        else:
            vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
            sd = comfy.utils.load_torch_file(vae_path)
            vae = comfy.sd.VAE(sd=sd)

        logger.info(f"Flux {base_type} stack loaded successfully.")
        return (model, clip, vae)

    def _apply_flux_sampling(self, model):
        try:
            sampling = model.model.model_sampling
            if not hasattr(sampling, "shift"):
                logger.info("Applying Flux Sampling Shift (1.15).")
                sampling.set_parameters(shift=1.15)
        except Exception as e:
            logger.warning(f"Could not patch sampling: {e}")

    def _load_taesd(self, name):
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        prefixes = {"taesd": ["taesd_encoder.", "taesd_decoder."], "taesdxl": ["taesdxl_encoder.", "taesdxl_decoder."], "taesd3": ["taesd3_encoder.", "taesd3_decoder."], "taef1": ["taef1_encoder.", "taef1_decoder."]}
        enc_name = next(v for v in approx_vaes if v.startswith(prefixes[name][0]))
        dec_name = next(v for v in approx_vaes if v.startswith(prefixes[name][1]))
        enc_sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", enc_name))
        dec_sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", dec_name))
        sd = {}
        for k, v in enc_sd.items(): sd[f"taesd_encoder.{k}"] = v
        for k, v in dec_sd.items(): sd[f"taesd_decoder.{k}"] = v
        scales = {"taesd": (0.18215, 0.0), "taesdxl": (0.13025, 0.0), "taesd3": (1.5305, 0.0609), "taef1": (0.3611, 0.1159)}
        sd["vae_scale"], sd["vae_shift"] = torch.tensor(scales[name][0]), torch.tensor(scales[name][1])
        return comfy.sd.VAE(sd=sd)

# Registration info (referenced in __init__.py)
# NODE_CLASS_MAPPINGS = { "FluxGGUFLoader": FluxGGUFLoader }
# NODE_DISPLAY_NAME_MAPPINGS = { "FluxGGUFLoader": "Flux GGUF Advanced Loader" }
