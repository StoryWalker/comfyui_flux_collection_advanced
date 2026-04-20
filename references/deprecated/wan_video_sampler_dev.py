# -*- coding: utf-8 -*-
import torch
import logging
import os
import re
import nodes
import node_helpers
import comfy.utils
import comfy.model_management
import comfy.sample
from typing import Any, Dict, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WanVideoSampler_Dev:
    """
    [DEV] v1.20 Style-Integrated Sampler for Wan 2.2.
    Features: 
    - 2 Style Selectors (from styles.csv) integrated into the prompt flow.
    - Extreme VRAM management for long storytelling chains.
    - Automatic last frame extraction and component passthrough.
    """
    
    # --- Style Loading Logic (Mirrored from FluxTextPrompt) ---
    _cached_styles = {}
    _styles_loaded = False
    _STYLES_FILENAME = "styles.csv"

    @classmethod
    def _load_styles(cls):
        if not cls._styles_loaded:
            # styles.csv vive en la raiz del plugin, subimos un nivel desde pending_refactor/
            styles_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), cls._STYLES_FILENAME)
            if os.path.exists(styles_path):
                try:
                    with open(styles_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()[1:]
                        csv_regex = re.compile(r',(?=(?:[^"]*"[^"]*")*[^"]*$)')
                        for line in lines:
                            if not line.strip(): continue
                            parts = [p.strip().strip('"') for p in csv_regex.split(line)]
                            if len(parts) >= 3:
                                cls._cached_styles[parts[0]] = [parts[1], parts[2]]
                    cls._styles_loaded = True
                    logger.info(f"[DEV] Loaded {len(cls._cached_styles)} styles for Wan Sampler.")
                except Exception as e:
                    logger.error(f"[DEV] Failed to load styles.csv: {e}")
            else:
                cls._cached_styles = {"None": ["", ""]}
                cls._styles_loaded = True

    @classmethod
    def INPUT_TYPES(cls):
        cls._load_styles()
        # Remove sorted() to keep original CSV order
        style_names = ["None"] + list(cls._cached_styles.keys())
        
        return {
            "required": {
                "reference_image": ("IMAGE", {"tooltip": "Starting frame."}),
                "model_high": ("MODEL",),
                "model_low": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "clip_vision": ("CLIP_VISION",),
                "positive_prompt": ("STRING", {"multiline": True, "default": "Cinematic video"}),
                "style_1": (style_names, {"default": "None"}),
                "style_2": (style_names, {"default": "None"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "worst quality"}),
                "width": ("INT", {"default": 480, "min": 16, "max": 2048, "step": 16}),
                "height": ("INT", {"default": 848, "min": 16, "max": 2048, "step": 16}),
                "num_frames": ("INT", {"default": 81, "min": 1, "max": 241, "step": 4}),
                "steps_high": ("INT", {"default": 4, "min": 1, "max": 50}),
                "steps_low": ("INT", {"default": 4, "min": 1, "max": 50}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "upscale_method": (["nearest-exact", "bilinear", "bicubic", "area", "lanczos"], {"default": "nearest-exact"}),
                "crop_position": (["center", "top", "bottom", "left", "right"], {"default": "center"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "ANY", "CLIP", "VAE", "CLIP_VISION",)
    RETURN_NAMES = ("image", "last_image", "any", "clip", "vae", "clip_vision",)
    FUNCTION = "generate_video"
    CATEGORY = "flux_collection_advanced/_dev"

    def _get_full_prompt(self, base_text, s1, s2, is_positive=True):
        idx = 0 if is_positive else 1
        prompts = [base_text]
        for s in [s1, s2]:
            if s != "None" and s in self._cached_styles:
                style_text = self._cached_styles[s][idx]
                if style_text: prompts.append(style_text)
        return ", ".join([p for p in prompts if p.strip()])

    def _encode_prompt(self, clip, prompt):
        tokens = clip.tokenize(prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]

    def generate_video(self, **kwargs):
        # 1. Extraction
        m_high, m_low, clip, vae, cv = kwargs.get("model_high"), kwargs.get("model_low"), kwargs.get("clip"), kwargs.get("vae"), kwargs.get("clip_vision")
        p_base, n_base = kwargs.get("positive_prompt"), kwargs.get("negative_prompt")
        s1, s2 = kwargs.get("style_1"), kwargs.get("style_2")
        w, h, f = kwargs.get("width"), kwargs.get("height"), kwargs.get("num_frames")
        s_high, s_low, cfg, seed, dns = kwargs.get("steps_high"), kwargs.get("steps_low"), kwargs.get("cfg"), kwargs.get("seed"), kwargs.get("denoise")
        
        device = comfy.model_management.get_torch_device()

        # 2. Build Styled Prompts
        logger.info(f"[DEV] Applying Styles: {s1}, {s2}")
        full_pos = self._get_full_prompt(p_base, s1, s2, True)
        full_neg = self._get_full_prompt(n_base, s1, s2, False)
        
        # 3. Encoding and Prep
        pos = self._encode_prompt(clip, full_pos)
        neg = self._encode_prompt(clip, full_neg)
        
        img_in = kwargs.get("reference_image").movedim(-1, 1)
        img_res = comfy.utils.common_upscale(img_in, w, h, kwargs.get("upscale_method"), kwargs.get("crop_position")).movedim(1, -1)
        cv_out = cv.encode_image(self._center_crop(img_res), crop=False)

        # 4. Context
        latent_t = ((f - 1) // 4) + 1
        latent = torch.zeros([1, 16, latent_t, h // 8, w // 8], device=device)
        sequence = torch.ones((f, h, w, 3), device=device, dtype=img_res.dtype) * 0.5
        sequence[0] = img_res[0]
        concat_img = vae.encode(sequence)
        concat_mask = torch.ones((1, 1, latent_t, h // 8, w // 8), device=device); concat_mask[:, :, :1] = 0.0 
        
        c_vals = {"clip_vision_output": cv_out, "concat_latent_image": concat_img, "concat_mask": concat_mask}
        p_final = node_helpers.conditioning_set_values(pos, c_vals)
        n_final = node_helpers.conditioning_set_values(neg, c_vals)

        # 5. Dual Sampling
        l_dict = {"samples": latent}
        samples = nodes.common_ksampler(m_high, seed, s_high+s_low, cfg, "lcm", "simple", p_final, n_final, l_dict, dns, False, 0, s_high, False)[0]
        comfy.model_management.soft_empty_cache()
        samples = nodes.common_ksampler(m_low, seed, s_high+s_low, cfg, "lcm", "simple", p_final, n_final, samples, dns, True, s_high, s_high+s_low, True)[0]

        # 6. Final Sync
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        v_out = vae.decode(samples["samples"])
        if len(v_out.shape) == 5 and v_out.shape[0] == 1: v_out = v_out.squeeze(0)
        
        return (v_out, v_out[-1:].clone(), None, clip, vae, cv)

    def _center_crop(self, image):
        h, w = image.shape[1:3]; length = min(h, w)
        y, x = (h - length) // 2, (w - length) // 2
        return image[:, y:y+length, x:x+length, :]

# Registered via __init__.py
