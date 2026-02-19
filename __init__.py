# Imports for your Python node classes
from custom_nodes.comfyui_flux_collection_advanced.flux_models_loader import FluxModelsLoader
from custom_nodes.comfyui_flux_collection_advanced.flux_gguf_loader import FluxGGUFLoader
from custom_nodes.comfyui_flux_collection_advanced.flux_text_prompt import FluxTextPrompt
from custom_nodes.comfyui_flux_collection_advanced.flux_sampler_parameters import FluxSamplerParameters
from custom_nodes.comfyui_flux_collection_advanced.flux_controlnet_loader import FluxControlNetLoader
from custom_nodes.comfyui_flux_collection_advanced.flux_controlnet_apply import FluxControlNetApply
from custom_nodes.comfyui_flux_collection_advanced.flux_controlnet_apply_preview import FluxControlNetApplyPreview
from custom_nodes.comfyui_flux_collection_advanced.flux_image_preview import FluxImagePreview
from custom_nodes.comfyui_flux_collection_advanced.flux_image_upscaler import FluxImageUpscaler
from custom_nodes.comfyui_flux_collection_advanced.flux_vram_loader_beta import FluxModelsLoader_VRAM_Beta

# --- Version Information ---
__version__ = "0.2.0" 

# --- Added Line ---
# Tells ComfyUI to serve static files (like JS) from the 'js' folder
# Ensure this folder exists: comfyui_flux_collection_advanced/js/
WEB_DIRECTORY = "./js"
# --------------------

NODE_CLASS_MAPPINGS = {
    "FluxLoader": FluxModelsLoader,
    "FluxGGUFLoader": FluxGGUFLoader,
    "FluxTextPrompt": FluxTextPrompt,
    "FluxSamplerParameters": FluxSamplerParameters,
    "FluxControlNetLoader": FluxControlNetLoader,
    "FluxControlNetApply": FluxControlNetApply,
    "FluxControlNetApplyPreview": FluxControlNetApplyPreview,
    "FluxImagePreview": FluxImagePreview,
    "FluxImageUpscaler": FluxImageUpscaler, 
    "FluxVRAMLoaderBeta": FluxModelsLoader_VRAM_Beta,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLoader": "Flux Models Loader (Advanced)",
    "FluxGGUFLoader": "Flux GGUF Advanced Loader",
    "FluxTextPrompt": "Flux Text Prompt Styler (4 Styles)",
    "FluxSamplerParameters": "Flux Generate, Sample & Decode",
    "FluxControlNetLoader": "Flux ControlNet Loader",
    "FluxControlNetApply": "Flux ControlNet Apply",
    "FluxControlNetApplyPreview": "Flux ControlNet Apply + Preview",
    "FluxImagePreview": "Flux Image Preview (Advanced)", 
    "FluxImageUpscaler": "Flux Image Upscaler (Spandrel)", 
    "FluxVRAMLoaderBeta": "Flux VRAM Loader (BETA/Optimization)",
}

# Optional: Message to confirm nodes loaded (useful for debugging)
print("------------------------------------------")
print(f"Flux Collection Advanced Nodes Loaded - Version: {__version__}")
print("------------------------------------------")