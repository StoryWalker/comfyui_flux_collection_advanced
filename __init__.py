# Imports for your Python node classes
from custom_nodes.comfyui_flux_collection_advanced.flux_models_loader import FluxModelsLoader
from custom_nodes.comfyui_flux_collection_advanced.flux_text_prompt import FluxTextPrompt
from custom_nodes.comfyui_flux_collection_advanced.flux_sampler_parameters import FluxSamplerParameters
from custom_nodes.comfyui_flux_collection_advanced.flux_controlnet_loader import FluxControlNetLoader
from custom_nodes.comfyui_flux_collection_advanced.flux_controlnet_apply import FluxControlNetApply
from custom_nodes.comfyui_flux_collection_advanced.flux_controlnet_apply_preview import FluxControlNetApplyPreview
from custom_nodes.comfyui_flux_collection_advanced.flux_image_preview import FluxImagePreview

# --- Added Line ---
# Tells ComfyUI to serve static files (like JS) from the 'js' folder
# Ensure this folder exists: comfyui_flux_collection_advanced/js/
WEB_DIRECTORY = "./js"
# --------------------

NODE_CLASS_MAPPINGS = {
    "FluxLoader": FluxModelsLoader,
    "FluxTextPrompt": FluxTextPrompt,
    "FluxSamplerParameters": FluxSamplerParameters,
    "FluxControlNetLoader": FluxControlNetLoader,
    "FluxControlNetApply": FluxControlNetApply,
    "FluxControlNetApplyPreview": FluxControlNetApplyPreview,
    "FluxImagePreview": FluxImagePreview,
    #"FluxImageUpscaler": FluxImageUpscaler, # Keeping commented out as before
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLoader": "FluxLoader",
    "FluxTextPrompt": "FluxTextPrompt",
    "FluxSamplerParameters": "FluxSamplerParameters",
    "FluxControlNetLoader": "FluxControlNetLoader",
    "FluxControlNetApply": "FluxControlNetApply",
    "FluxControlNetApplyPreview": "FluxControlNetApplyPreview",
    "FluxImagePreview": "FluxImagePreview", 
    #"FluxImageUpscaler": "FluxImageUpscaler", # Keeping commented out as before
}

# Optional: Message to confirm nodes loaded (useful for debugging)
print("------------------------------------------")
print("Flux Collection Advanced Nodes Loaded")
print("------------------------------------------")