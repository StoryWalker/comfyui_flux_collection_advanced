# /application/infrastructure/comfyui_adapters/flux_image_save_adapter.py
__version__ = "1.4.1"
# 1.4.1 - Corrected UI implementation to display output text and removed node output socket.
# 1.4.0 - Reordered inputs and added multiline text output for save paths, removed node output.
# 1.3.0 - Added text output for save path and split subfolder input into two fields.
# 1.2.0 - Reordered inputs and made subfolder an editable combo box.
# 1.1.0 - Added subfolder selection widget.
# 1.0.0 - Initial version of FluxImageSave node.

import logging
from typing import Any, Dict, List

from infrastructure.dependency_injection.container import Container
from domain.entities import Image, ImageMetadata, ImageSaveConfig

logger = logging.getLogger(__name__)

class FluxImageSave:
    """
    Adapter for the SaveImage node, following hexagonal architecture.
    """
    container = Container()

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Defines the input types for the node."""
        repo = cls.container.image_repository()
        subfolders = repo.get_available_output_subfolders()

        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to save."}),
                # El orden correcto que solicitaste
                "subfolder_name": (subfolders, {"tooltip": "Select an existing subfolder."}),
                "custom_subfolder": ("STRING", {"default": "", "tooltip": "Type a new folder name here. Overrides the selection above."}),
                "filename_prefix": ("STRING", {"default": "flux_output"}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    # CORRECCIÓN: Nos aseguramos de que no haya salidas de conector
    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "flux_collection_advanced/refactored"

    def save_images(
        self, images: Any, subfolder_name: str, filename_prefix: str,
        custom_subfolder: str = "", prompt: Dict = None, extra_pnginfo: Dict = None
    ) -> Dict:
        """Executes the refactored image saving logic."""
        node_name = self.__class__.__name__
        logger.info(f"Executing node: {node_name}")

        try:
            final_subfolder = custom_subfolder.strip() if custom_subfolder.strip() else subfolder_name

            image_list: List[Image] = [Image(data=img) for img in images]
            metadata = ImageMetadata(prompt=prompt, extra_info=extra_pnginfo)
            save_config = ImageSaveConfig(
                filename_prefix=filename_prefix,
                metadata=metadata,
                subfolder=final_subfolder
            )

            use_case = self.container.save_images_use_case()
            results = use_case.execute(image_list, save_config)

            # Preparamos los resultados para la previsualización de la imagen
            ui_results = [
                {"filename": r.filename, "subfolder": r.subfolder, "type": r.type}
                for r in results
            ]

            # Construimos la lista de strings con las rutas completas
            all_paths = [r.full_path for r in results]

            # CORRECCIÓN: Devolvemos el texto a través de la clave "text" en el diccionario "ui"
            # Esto mostrará el texto directamente en el nodo.
            return {"ui": {"images": ui_results, "text": all_paths}}

        except Exception as e:
            logger.exception(f"A critical error occurred in {node_name}: {e}")
            raise