# /custom_nodes/comfyui_flux_collection_advanced/__init__.py
__version__ = "2.5.0"
# 2.5.0 - Added new refactored node FluxImageSave.
# 2.4.2 - Added VAE repository to diagnostics.
# 2.4.1 - Centralized color codes into a utility module.
# 2.4.0 - Added blue and yellow to the colorized console diagnostics.
# 2.3.1 - Added colorized console diagnostics for better error visibility.
# 2.3.0 - Added refactored SamplerParameters node.
# 2.2.0 - Added version diagnostics for all refactored modules.
# 2.1.0 - Added refactored TextPrompt node.
# 2.0.0 - Implemented self-contained hexagonal architecture support for refactoring.

import sys
import os
import importlib

# --- Path setup ---
try:
    current_dir = os.path.dirname(__file__)
    application_path = os.path.join(current_dir, 'application')
    if os.path.exists(application_path) and application_path not in sys.path:
        sys.path.insert(0, application_path)
except Exception:
    pass # Fail silently if something goes wrong with pathing

# --- ANSI Color Codes for Console Output ---
try:
    from infrastructure.utils.logging_colors import (
        COLOR_BLUE, COLOR_GREEN, COLOR_RED, COLOR_YELLOW, COLOR_RESET
    )
except ImportError:
    # Fallback to empty strings if colors can't be imported, so the script doesn't crash.
    COLOR_BLUE = COLOR_GREEN = COLOR_RED = COLOR_YELLOW = COLOR_RESET = ""

print(f"{COLOR_BLUE}--- Initializing 'comfyui_flux_collection_advanced' (v{__version__} with diagnostics) ---{COLOR_RESET}")

# --- Version Diagnostics with Colors ---
print(f"\n{COLOR_BLUE}[*] --- Verifying loaded module versions ---{COLOR_RESET}")
modules_to_check = [
    "domain.entities",
    "domain.ports",
    "domain.use_cases",
    "infrastructure.repositories.comfy_model_repository",
    "infrastructure.repositories.comfy_prompt_encoder_repository",
    "infrastructure.repositories.comfy_sampler_repository",
    "infrastructure.repositories.comfy_vae_repository",
    "infrastructure.repositories.comfy_image_repository",
    "infrastructure.dependency_injection.container",
    "infrastructure.comfyui_adapters.flux_models_loader_adapter",
    "infrastructure.comfyui_adapters.flux_text_prompt_adapter",
    "infrastructure.comfyui_adapters.flux_sampler_parameters_adapter",
    "infrastructure.comfyui_adapters.flux_image_save_adapter",
]

all_modules_loaded = True
for module_name in modules_to_check:
    try:
        module = importlib.import_module(module_name)
        if not hasattr(module, '__version__'): module.__version__ = "1.0.0" # Default
        print(f"{COLOR_GREEN}[OK]   Loaded '{module_name}' (v{module.__version__}){COLOR_RESET}")
    except Exception as e:
        print(f"{COLOR_RED}[FAIL] Failed to import '{module_name}': {e}{COLOR_RESET}")
        all_modules_loaded = False

if all_modules_loaded:
    print(f"{COLOR_BLUE}[*] --- All refactored modules imported successfully ---{COLOR_RESET}")
else:
    print(f"{COLOR_YELLOW}[!] --- Some refactored modules failed to load, check errors above ---{COLOR_RESET}")

# --- Node Registration ---
if all_modules_loaded:
    from infrastructure.comfyui_adapters.flux_models_loader_adapter import FluxModelsLoaderRefactored
    from infrastructure.comfyui_adapters.flux_text_prompt_adapter import FluxTextPromptRefactored
    from infrastructure.comfyui_adapters.flux_sampler_parameters_adapter import FluxSamplerParametersRefactored
    from infrastructure.comfyui_adapters.flux_image_save_adapter import FluxImageSave
else:
    FluxModelsLoaderRefactored = None
    FluxTextPromptRefactored = None
    FluxSamplerParametersRefactored = None
    FluxImageSave = None

# Original Nodes
from .flux_models_loader import FluxModelsLoader
from .flux_text_prompt import FluxTextPrompt
from .flux_sampler_parameters import FluxSamplerParameters
from .flux_controlnet_loader import FluxControlNetLoader
from .flux_controlnet_apply import FluxControlNetApply
from .flux_controlnet_apply_preview import FluxControlNetApplyPreview
from .flux_image_preview import FluxImagePreview
from .flux_image_upscaler import FluxImageUpscaler

WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    "FluxLoader": FluxModelsLoader, "FluxTextPrompt": FluxTextPrompt,
    "FluxSamplerParameters": FluxSamplerParameters, "FluxControlNetLoader": FluxControlNetLoader,
    "FluxControlNetApply": FluxControlNetApply, "FluxControlNetApplyPreview": FluxControlNetApplyPreview,
    "FluxImagePreview": FluxImagePreview, "FluxImageUpscaler": FluxImageUpscaler,
}
if FluxModelsLoaderRefactored: NODE_CLASS_MAPPINGS["FluxModelsLoaderRefactored"] = FluxModelsLoaderRefactored
if FluxTextPromptRefactored: NODE_CLASS_MAPPINGS["FluxTextPromptRefactored"] = FluxTextPromptRefactored
if FluxSamplerParametersRefactored: NODE_CLASS_MAPPINGS["FluxSamplerParametersRefactored"] = FluxSamplerParametersRefactored
if FluxImageSave: NODE_CLASS_MAPPINGS["FluxImageSave"] = FluxImageSave


NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxLoader": "FluxLoader", "FluxTextPrompt": "FluxTextPrompt",
    "FluxSamplerParameters": "FluxSamplerParameters", "FluxControlNetLoader": "FluxControlNetLoader",
    "FluxControlNetApply": "FluxControlNetApply", "FluxControlNetApplyPreview": "FluxControlNetApplyPreview",
    "FluxImagePreview": "FluxImagePreview", "FluxImageUpscaler": "FluxImageUpscaler",
}
if FluxModelsLoaderRefactored: NODE_DISPLAY_NAME_MAPPINGS["FluxModelsLoaderRefactored"] = "Flux Models Loader (Refactored)"
if FluxTextPromptRefactored: NODE_DISPLAY_NAME_MAPPINGS["FluxTextPromptRefactored"] = "Flux Text Prompt (Refactored)"
if FluxSamplerParametersRefactored: NODE_DISPLAY_NAME_MAPPINGS["FluxSamplerParametersRefactored"] = "Flux Sampler & Decode (Refactored)" # Renamed for clarity
if FluxImageSave: NODE_DISPLAY_NAME_MAPPINGS["FluxImageSave"] = "Flux Image Save (Refactored)"


print("------------------------------------------")
print(f"{COLOR_BLUE}Flux Collection Advanced Nodes Loaded - Version: {__version__}{COLOR_RESET}")
print("------------------------------------------")