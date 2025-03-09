from comfyui_flux_collection_advanced.flux_models_loader import FluxModelsLoader
from comfyui_flux_collection_advanced.flux_text_prompt import FluxTextPrompt
#from comfyui_flux_collection_advanced.flux_image_upscaler import FluxImageUpscaler


NODE_CLASS_MAPPINGS = {
    "FluxLoader": FluxModelsLoader,
    "FluxTextPrompt": FluxTextPrompt,
    #"FluxImageUpscaler": FluxImageUpscaler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLoader": "FluxLoader",
    "FluxTextPrompt": "FluxTextPrompt",
    #"FluxImageUpscaler": "FluxImageUpscaler",
}