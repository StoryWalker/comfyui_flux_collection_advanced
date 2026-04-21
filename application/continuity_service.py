# Task-Source: T#4
# -*- coding: utf-8 -*-
import torch
import logging
try:
    from domain.models import BufferConfig
    from infrastructure.persistence_adapter import FilePersistenceAdapter
except ImportError:
    from ..domain.models import BufferConfig
    from ..infrastructure.persistence_adapter import FilePersistenceAdapter

logger = logging.getLogger(__name__)

class ContinuityService:
    """
    Application Service:
    Gestiona la logica de conmutacion entre frame inicial y frame persistente en disco.
    """

    def __init__(self, persistence=None):
        self.persistence = persistence if persistence is not None else FilePersistenceAdapter()

    def sync_image(self, config: BufferConfig, initial_image=None, sampler_last_image=None,
                   fallback_height: int = 480, fallback_width: int = 848):
        """
        Sincroniza el frame de salida segun el modo del buffer.

        Args:
            fallback_height: Altura del tensor de fallback cuando no hay imagen disponible.
            fallback_width:  Anchura del tensor de fallback cuando no hay imagen disponible.
                             Deben derivarse de la resolucion configurada en el nodo.
        """
        # 1. Actualizar disco si llego un frame nuevo del sampler
        if sampler_last_image is not None:
            self.persistence.save_frame(sampler_last_image)

        # 2. Logica de salida segun modo
        if config.is_initial:
            if initial_image is not None:
                # Guardar inicial en disco para preparar el siguiente paso
                self.persistence.save_frame(initial_image)
                return initial_image
            # Sin imagen inicial: devolver tensor vacio con dimensiones configuradas
            logger.warning(f"[HEX] Continuity: Sin imagen inicial. Generando tensor vacio ({fallback_height}x{fallback_width}).")
            return torch.zeros((1, fallback_height, fallback_width, 3))

        # 3. Cargar desde disco
        persistent_frame = self.persistence.load_frame()
        if persistent_frame is not None:
            return persistent_frame

        # Fallback final: no hay frame en disco ni imagen inicial
        logger.error(f"[HEX] Continuity Error: No se encontro ningun frame. Devolviendo tensor vacio ({fallback_height}x{fallback_width}).")
        return torch.zeros((1, fallback_height, fallback_width, 3))
