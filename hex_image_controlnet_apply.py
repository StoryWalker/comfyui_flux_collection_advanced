# Task-Source: T#16
# -*- coding: utf-8 -*-
import logging
import torch
from .infrastructure.error_adapter import hex_error_handler

logger = logging.getLogger(__name__)


class FluxControlNetApplyHex:
    """[HEX] ControlNet Apply — aplica ControlNet al conditioning positivo (Flux)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "section_config": ("STRING", {"default": "CONTROLNET APPLY"}),
                "positive":       ("CONDITIONING",),
                "control_net":    ("CONTROL_NET",),
                "image":          ("IMAGE",),
                "strength":       ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent":  ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0,  "step": 0.001}),
                "end_percent":    ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0,  "step": 0.001}),
            },
            "optional": {
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES  = ("CONDITIONING",)
    RETURN_NAMES  = ("positive",)
    FUNCTION      = "execute"
    CATEGORY      = "flux_collection_advanced/hex"
    DESCRIPTION = (
        "[HEX] ControlNet Apply — aplica ControlNet al conditioning positivo (Flux).\n\n"
        "This documentation was AI-generated. If you find any errors or have suggestions for "
        "improvement, please feel free to contribute! Edit on GitHub."
    )

    @hex_error_handler
    def execute(self, **kwargs):
        positive      = kwargs["positive"]
        control_net   = kwargs["control_net"]
        image         = kwargs["image"]
        strength      = kwargs["strength"]
        start_percent = kwargs["start_percent"]
        end_percent   = kwargs["end_percent"]
        vae           = kwargs.get("vae", None)

        if strength == 0:
            return (positive,)

        if image.ndim != 4:
            raise ValueError(f"[HEX] ControlNet Apply: imagen debe ser 4D (BHWC), recibida {image.ndim}D.")
        if start_percent > end_percent:
            raise ValueError(f"[HEX] ControlNet Apply: start_percent ({start_percent}) > end_percent ({end_percent}).")

        control_hint = image.movedim(-1, 1)
        timing = (start_percent, end_percent)
        cache  = {}

        processed = []
        for tensor_data, cond_dict in positive:
            d    = cond_dict.copy()
            prev = d.get("control", None)
            if prev not in cache:
                cn = control_net.copy()
                cn = cn.set_cond_hint(control_hint, strength, timing, vae=vae)
                cn.set_previous_controlnet(prev)
                cache[prev] = cn
            d["control"]               = cache[prev]
            d["control_apply_to_uncond"] = False
            processed.append([tensor_data, d])

        logger.info(f"[HEX] ControlNet Apply: strength={strength}, range=[{start_percent}, {end_percent}]")
        return (processed,)

# Registrado via __init__.py
