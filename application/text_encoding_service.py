# -*- coding: utf-8 -*-
"""
Servicio de aplicacion para la codificacion de texto con estilos para nodos Flux.
Orquesta la carga de estilos CSV, construccion del prompt y codificacion CLIP.
"""
import logging
import re
from typing import Any, Dict, List

from ..domain.models import TextEncodingConfig
from ..infrastructure.clip_encoding_adapter import ClipEncodingAdapter

logger = logging.getLogger(__name__)

_NO_STYLE_KEY = "No Style"


class TextEncodingService:
    """
    Servicio de aplicacion que gestiona la carga de estilos y la codificacion
    de prompts. Los estilos se cachean a nivel de clase para evitar releer el
    CSV en cada ejecucion del nodo.
    """

    _styles: Dict[str, List[str]] = {}
    _loaded: bool = False

    def __init__(self, clip_adapter: ClipEncodingAdapter):
        self.clip_adapter = clip_adapter

    # ------------------------------------------------------------------
    # Gestion de estilos (clase)
    # ------------------------------------------------------------------

    @classmethod
    def load_styles(cls, styles_path: str) -> None:
        """
        Carga estilos desde el CSV si todavia no se han cargado.
        Hilo seguro para el caso de uso de ComfyUI (single-process).
        """
        if cls._loaded:
            return

        styles: Dict[str, List[str]] = {_NO_STYLE_KEY: ["", ""]}
        csv_regex = re.compile(r',(?=(?:[^"]*"[^"]*")*[^"]*$)')

        try:
            with open(styles_path, "r", encoding="utf-8") as f:
                lines = f.readlines()[1:]  # omitir cabecera

            for i, line in enumerate(lines):
                if not line.strip():
                    continue
                parts = [p.strip().strip('"') for p in csv_regex.split(line)]
                if len(parts) >= 3 and parts[0]:
                    styles[parts[0]] = [parts[1], parts[2]]
                else:
                    logger.warning(f"[TextEncodingService] Fila {i+2} malformada, omitida.")

            logger.info(f"[TextEncodingService] {len(styles)} estilos cargados desde {styles_path}")

        except FileNotFoundError:
            logger.error(f"[TextEncodingService] styles.csv no encontrado en: {styles_path}")
        except Exception as e:
            logger.exception(f"[TextEncodingService] Error leyendo styles.csv: {e}")

        cls._styles = styles
        cls._loaded = True

    @classmethod
    def get_style_names(cls) -> List[str]:
        return list(cls._styles.keys()) if cls._styles else [_NO_STYLE_KEY]

    # ------------------------------------------------------------------
    # Codificacion
    # ------------------------------------------------------------------

    def encode(self, config: TextEncodingConfig, clip: Any) -> Any:
        """
        Construye el prompt estilizado y lo codifica con CLIP.

        Returns:
            conditioning tensor.
        """
        styled_text = config.get_styled_prompt(self._styles)
        if not styled_text:
            logger.warning("[TextEncodingService] Prompt vacio tras combinar texto y estilos.")
        return self.clip_adapter.encode(clip, styled_text, config.guidance)
