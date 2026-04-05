# -*- coding: utf-8 -*-
"""
Adaptador de infraestructura para tokenizacion, codificacion CLIP
y aplicacion de guidance para nodos Flux.
"""
import logging
import node_helpers
from typing import Any

logger = logging.getLogger(__name__)


class ClipEncodingAdapter:
    """
    Encapsula toda interaccion con el objeto CLIP de ComfyUI
    y con node_helpers para aplicar guidance.
    """

    @staticmethod
    def encode(clip: Any, text: str, guidance: float) -> Any:
        """
        Tokeniza el texto, lo codifica con CLIP y aplica el guidance scale.

        Returns:
            conditioning tensor listo para el sampler.
        """
        logger.info(f"[Flux] Codificando prompt ({len(text)} chars), guidance={guidance}")

        if not hasattr(clip, "tokenize"):
            raise RuntimeError("El objeto CLIP no tiene metodo 'tokenize'.")
        if not hasattr(clip, "encode_from_tokens"):
            raise RuntimeError("El objeto CLIP no tiene metodo 'encode_from_tokens'.")

        tokens = clip.tokenize(text)
        output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        conditioning = [[cond, output]]
        return node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})
