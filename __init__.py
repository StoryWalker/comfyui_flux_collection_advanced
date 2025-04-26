from custom_nodes.comfyui_flux_collection_advanced.flux_models_loader import FluxModelsLoader
from custom_nodes.comfyui_flux_collection_advanced.flux_text_prompt import FluxTextPrompt
from custom_nodes.comfyui_flux_collection_advanced.flux_sampler_parameters import FluxSamplerParameters
from custom_nodes.comfyui_flux_collection_advanced.flux_load_controlnet import FluxLoadControlNetPreprocessor
#from comfyui_flux_collection_advanced.flux_image_upscaler import FluxImageUpscaler


NODE_CLASS_MAPPINGS = {
    "FluxLoader": FluxModelsLoader,
    "FluxTextPrompt": FluxTextPrompt,
    "FluxSamplerParameters": FluxSamplerParameters,
    "FluxLoadControlNetPreprocessor": FluxLoadControlNetPreprocessor,
    #"FluxImageUpscaler": FluxImageUpscaler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLoader": "FluxLoader",
    "FluxTextPrompt": "FluxTextPrompt",
    "FluxSamplerParameters": "FluxSamplerParameters",
    "FluxLoadControlNetPreprocessor": "FluxLoadControlNetPreprocessor",
    #"FluxImageUpscaler": "FluxImageUpscaler",
}