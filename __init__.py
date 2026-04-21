# Task-Source: T#5
import logging
import os

# Configuración de colores ANSI para la consola
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

__version__ = "0.3.0"
WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

print(f"\n{BOLD}--- Flux Collection Advanced v{__version__} ---{RESET}")

def load_node(module_name, class_name, mapping_name, display_name):
    global NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    try:
        from importlib import import_module
        module = import_module(f".{module_name}", package=__name__)
        node_class = getattr(module, class_name)
        NODE_CLASS_MAPPINGS[mapping_name] = node_class
        NODE_DISPLAY_NAME_MAPPINGS[mapping_name] = display_name
        print(f"{GREEN}[OK]{RESET} {display_name}")
        return True
    except Exception as e:
        import traceback
        print(f"{RED}[ERROR]{RESET} {display_name}: {e}")
        traceback.print_exc()
        return False

# Registro secuencial de nodos [HEX] v2.4
load_node("hex_wan_unified_loader", "WanUnifiedLoaderHex", "WanUnifiedLoaderHex", "[HEX] Wan Unified Loader")
load_node("hex_wan_story_sampler", "WanStorySamplerHex", "WanStorySamplerHex", "[HEX] Wan Story Sampler")
load_node("hex_image_loader", "ImageLoaderHex", "ImageLoaderHex", "[HEX] Image Loader")
load_node("hex_loop_storage", "LoopStorageHex", "LoopStorageHex", "[HEX] Loop Storage")
load_node("hex_loop_fetcher", "LoopFetcherHex", "LoopFetcherHex", "[HEX] Loop Fetcher")
load_node("hex_prompt_sequencer", "PromptSequencerHex", "PromptSequencerHex", "[HEX] Prompt Sequencer")
load_node("hex_wan_video_saver", "WanVideoSaverHex", "WanVideoSaverHex", "[HEX] Wan Video Saver")
load_node("hex_image_models_loader", "FluxGGUFLoaderHex", "FluxGGUFLoader", "[HEX] Flux GGUF Loader")
load_node("hex_image_text_prompt", "FluxTextPromptHex", "FluxTextPromptHex", "[HEX] Flux Text Prompt")
load_node("hex_image_sampler_parameters", "FluxSamplerParametersHex", "FluxSamplerParametersHex", "[HEX] Flux Sampler Parameters")
load_node("hex_image_comparison", "ImageComparisonHex", "ImageComparisonHex", "[HEX] Image Comparison")
load_node("hex_global_seed", "GlobalSeedHex", "GlobalSeedHex", "[HEX] Global Seed")
load_node("hex_image_controlnet_loader", "FluxControlNetLoaderHex", "FluxControlNetLoaderHex", "[HEX] Flux ControlNet Loader")
load_node("hex_image_controlnet_apply", "FluxControlNetApplyHex", "FluxControlNetApplyHex", "[HEX] Flux ControlNet Apply")
load_node("hex_image_preview", "FluxImagePreviewHex", "FluxImagePreviewHex", "[HEX] Flux Image Preview")

# Registro de nodos de apoyo y DEV (solo con COMFYUI_DEV_NODES=1)
if os.getenv("COMFYUI_DEV_NODES", "0") == "1":
    load_node("test_hex_loaders", "Test_Hex_UnetLoader", "TestHexUnetLoader", "[TEST] Hex UNET Loader")
    load_node("test_hex_loaders", "Test_Hex_ClipLoader", "TestHexClipLoader", "[TEST] Hex CLIP Loader")
    load_node("test_hex_loaders", "Test_Hex_VaeLoader", "TestHexVaeLoader", "[TEST] Hex VAE Loader")
    load_node("test_hex_loaders", "Test_Hex_ClipVisionLoader", "TestHexClipVisionLoader", "[TEST] Hex CLIP Vision Loader")
    load_node("test_hex_loaders", "Test_Hex_LoraLoader", "TestHexLoraLoader", "[TEST] Hex LoRA Loader")
    load_node("test_hex_loaders", "Test_Hex_ModelSampling", "TestHexModelSampling", "[TEST] Hex Model Sampling")
    load_node("pending_refactor.wan_index_bridge_dev", "WanIndexBridge_Dev", "WanIndexBridgeDev", "[DEV] Wan Index Bridge")
    load_node("pending_refactor.wan_video_saver_dev", "WanVideoSaver_Dev", "WanVideoSaverDev", "[DEV] Wan Video Saver")
    print(f"{YELLOW}[DEV] Modo desarrollo activo — nodos TEST y DEV registrados.{RESET}")
else:
    print(f"[INFO] Nodos DEV/TEST omitidos. Activa con COMFYUI_DEV_NODES=1 para cargarlos.")

# Registro de nodos legacy Flux [pending_refactor]
# Flux Models Loader eliminado — reemplazado por hex_image_models_loader.py (HEX)
# Flux GGUF Loader migrado a hex_image_models_loader.py (HEX)
# Flux Text Prompt migrado a hex_image_text_prompt.py (HEX)
# Flux Sampler Parameters migrado a hex_image_sampler_parameters.py (HEX)
# Migrados a arquitectura hexagonal: T#15, T#16, T#17
# load_node("pending_refactor.flux_controlnet_loader", ...)   → hex_image_controlnet_loader.py
# load_node("pending_refactor.flux_controlnet_apply", ...)    → hex_image_controlnet_apply.py
# load_node("pending_refactor.flux_image_preview", ...)       → hex_image_preview.py
load_node("pending_refactor.flux_controlnet_apply_preview", "FluxControlNetApplyPreview", "FluxControlNetApplyPreview", "Flux ControlNet Apply Preview")
# Flux Image Comparison migrado a hex_image_comparison.py (HEX)
load_node("pending_refactor.flux_image_upscaler", "FluxImageUpscaler", "FluxImageUpscaler", "Flux Image Upscaler")
load_node("pending_refactor.flux_lora_detailer", "FluxLoraDetailer", "FluxLoraDetailer", "Flux Lora Detailer")
load_node("pending_refactor.flux_vram_loader_beta", "FluxModelsLoader_VRAM_Beta", "FluxVRAMLoaderBeta", "Flux VRAM Loader Beta")

print(f"{BOLD}Total nodos registrados: {len(NODE_CLASS_MAPPINGS)}{RESET}\n")
