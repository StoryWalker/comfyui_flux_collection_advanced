# /application/infrastructure/comfyui_adapters/flux_text_prompt_adapter.py
__version__ = "1.0.0"
# 1.0.0 - Refactored version of FluxTextPrompt.

import logging
from typing import Any, Dict

from infrastructure.dependency_injection.container import Container
from domain.entities import PromptConfig

logger = logging.getLogger(__name__)

class FluxTextPromptRefactored:
    """
    Refactored adapter for the FluxTextPrompt node.
    It translates UI inputs into a PromptConfig object and uses the
    EncodeTextWithStylesUseCase to get the final conditioning.
    """
    container = Container()

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Defines the input types for the node."""
        logger.info("Setting up INPUT_TYPES for FluxTextPromptRefactored...")
        try:
            # Use the repository to get the list of style names for the UI dropdowns
            repo = cls.container.prompt_encoder_repository()
            styles = repo.get_available_styles()
            style_names = list(styles.keys())
        except Exception as e:
            logger.exception(f"Could not fetch styles for node UI: {e}")
            style_names = ["Error: Could not load styles"]
        
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "clip": ("CLIP",),
                "style1": (style_names,),
                "style2": (style_names,),
                "style3": (style_names,),
                "style4": (style_names,),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1})
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "execute_refactored"
    CATEGORY = "flux_collection_advanced/refactored"

    def execute_refactored(
        self, clip: Any, text: str, style1: str, style2: str, 
        style3: str, style4: str, guidance: float
    ) -> tuple:
        """Executes the refactored text encoding logic."""
        node_name = self.__class__.__name__
        logger.info(f"Executing node: {node_name}")
        
        try:
            # 1. Create the domain-specific configuration object
            prompt_config = PromptConfig(
                clip_model=clip,
                base_text=text,
                style_names=[s for s in [style1, style2, style3, style4] if s and "Error" not in s],
                guidance=guidance
            )

            # 2. Get the use case from the DI container
            encode_uc = self.container.encode_text_use_case()

            # 3. Execute the use case
            conditioning_result = encode_uc.execute(prompt_config)
            
            logger.info(f"{node_name} execution successful.")
            
            # 4. Return the result in the format ComfyUI expects
            return (conditioning_result.data,)

        except Exception as e:
            logger.exception(f"A critical error occurred in {node_name}: {e}")
            raise