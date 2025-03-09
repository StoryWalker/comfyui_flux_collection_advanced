from comfyui_flux_collection_advanced.flux_models_loader import FluxModelsLoader
#from comfyui_flux_collection_advanced.flux_image_upscaler import FluxImageUpscaler


NODE_CLASS_MAPPINGS = {
    "FluxLoader": FluxModelsLoader,
    #"FluxImageUpscaler": FluxImageUpscaler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLoader": "FluxLoader",
    #"FluxImageUpscaler": "FluxImageUpscaler",
}