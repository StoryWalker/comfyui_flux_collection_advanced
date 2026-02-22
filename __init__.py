import logging

# Configuración de colores ANSI para la consola
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"

__version__ = "0.2.2"
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
        print(f"{RED}[ERROR]{RESET} {display_name}: {e}")
        return False

# Registro secuencial de nodos
load_node("flux_models_loader", "FluxModelsLoader", "FluxModelsLoader", "Flux Models Loader")
load_node("flux_gguf_loader", "FluxGGUFLoader", "FluxGGUFLoader", "Flux GGUF Loader")
load_node("flux_text_prompt", "FluxTextPrompt", "FluxTextPrompt", "Flux Text Prompt")
load_node("flux_sampler_parameters", "FluxSamplerParameters", "FluxSamplerParameters", "Flux Sampler Parameters")
load_node("flux_controlnet_loader", "FluxControlNetLoader", "FluxControlNetLoader", "Flux ControlNet Loader")
load_node("flux_controlnet_apply", "FluxControlNetApply", "FluxControlNetApply", "Flux ControlNet Apply")
load_node("flux_controlnet_apply_preview", "FluxControlNetApplyPreview", "FluxControlNetApplyPreview", "Flux ControlNet Apply Preview")
load_node("flux_image_preview", "FluxImagePreview", "FluxImagePreview", "Flux Image Preview")
load_node("flux_image_comparison", "FluxImageComparison", "FluxImageComparison", "Flux Image Comparison")
load_node("flux_image_upscaler", "FluxImageUpscaler", "FluxImageUpscaler", "Flux Image Upscaler")
load_node("flux_lora_detailer", "FluxLoraDetailer", "FluxLoraDetailer", "Flux Lora Detailer")
load_node("flux_vram_loader_beta", "FluxModelsLoader_VRAM_Beta", "FluxVRAMLoaderBeta", "Flux VRAM Loader Beta")
load_node("wan_video_loader_dev", "WanVideoLoader_Dev", "WanVideoLoaderDev", "[DEV] Wan Video Loader")
load_node("wan_video_sampler_dev", "WanVideoSampler_Dev", "WanVideoSamplerDev", "[DEV] Wan Video Sampler")
load_node("wan_context_dev", "WanContext_Dev", "WanContextDev", "[DEV] Wan Context")
load_node("wan_dual_sampler_dev", "WanDualSampler_Dev", "WanDualSamplerDev", "[DEV] Wan Dual Sampler")
load_node("wan_loader_dev", "WanVideoLoader_Dev", "WanVideoLoaderDev", "[DEV] Wan Loader")
load_node("wan_scene_encoder_dev", "WanSceneEncoder_Dev", "WanSceneEncoderDev", "[DEV] Wan Scene Encoder")
load_node("wan_video_joiner_dev", "WanVideoJoiner_Dev", "WanVideoJoinerDev", "[DEV] Wan Video Joiner")
load_node("wan_history_dev", "WanContinuityBuffer_Dev", "WanContinuityBufferDev", "[DEV] Wan Continuity Buffer")
load_node("wan_history_dev", "WanPromptSequencer_Dev", "WanPromptSequencerDev", "[DEV] Wan Prompt Sequencer")
load_node("wan_prompt_encoder_dev", "WanPromptEncoder_Dev", "WanPromptEncoderDev", "[DEV] Wan Prompt Encoder")
load_node("wan_story_sampler_dev", "WanStorySampler_v2", "WanStorySampler_v2", "[DEV] Wan Story Sampler")
load_node("wan_index_bridge_dev", "WanIndexBridge_Dev", "WanIndexBridgeDev", "[DEV] Wan Index Bridge")
load_node("wan_video_saver_dev", "WanVideoSaver_Dev", "WanVideoSaverDev", "[DEV] Wan Video Saver")

print(f"{BOLD}Total nodos registrados: {len(NODE_CLASS_MAPPINGS)}{RESET}\n")
