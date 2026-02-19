"""
Flux GGUF Metadata Extractor
Version: 1.0.0
Date: 2026-02-19
Author: TEAM_PRO (via Gemini CLI)
Description: Extracts technical metadata from GGUF files (Architecture, Quantization, Resolution).
"""
import logging
import folder_paths
import nodes
from typing import Any, Dict, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FluxGGUFInfo(nodes.ComfyNodeABC):
    """
    Node to extract and display metadata from GGUF files.
    Helpful for identifying model types and training resolutions.
    """
    
    FUNCTION = "get_metadata"
    CATEGORY = "flux_collection_advanced"
    DESCRIPTION = "Extracts and displays technical metadata from GGUF files."
    RETURN_TYPES = ("STRING", "INT", "INT")
    RETURN_NAMES = ("metadata_report", "suggested_width", "suggested_height")

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        unet_list = folder_paths.get_filename_list("unet_gguf")
        if not unet_list:
            unet_list = folder_paths.get_filename_list("diffusion_models")
            
        return {
            "required": {
                "unet_name": (sorted(unet_list), {"tooltip": "Select a GGUF file to inspect."}),
            }
        }

    def get_metadata(self, unet_name: str) -> Tuple[str, int, int]:
        logger.info(f"Extracting metadata from: {unet_name}")
        
        # Resolve path
        path = folder_paths.get_full_path("unet_gguf", unet_name) or 
               folder_paths.get_full_path("diffusion_models", unet_name) or 
               folder_paths.get_full_path("unet", unet_name)
               
        if not path:
            return ("Error: File not found.", 1024, 1024)

        report = []
        width, height = 1024, 1024
        
        try:
            # We use the 'gguf' library which is a dependency of ComfyUI-GGUF
            from gguf import GGUFReader
            reader = GGUFReader(path)
            
            report.append(f"### GGUF Metadata Report: {unet_name}")
            report.append(f"- **File Path:** {path}")
            
            # 1. General Info
            arch = reader.get_field("general.architecture")
            if arch: report.append(f"- **Architecture:** {arch}")
            
            name = reader.get_field("general.name")
            if name: report.append(f"- **Model Name:** {name}")
            
            # 2. Quantization Info
            quant_version = reader.get_field("general.quantization_version")
            if quant_version: report.append(f"- **Quantization Version:** {quant_version}")
            
            # 3. Detect Resolution (if available in custom keys)
            # Some GGUF creators store the original shape or resolution
            orig_shape = reader.get_field("comfy.gguf.orig_shape")
            if orig_shape:
                report.append(f"- **Original Shape (Latent):** {orig_shape}")
                # For Flux/SDXL, usually [Batch, Channels, H/16, W/16]
                if len(orig_shape) >= 4:
                    height = int(orig_shape[2]) * 16
                    width = int(orig_shape[3]) * 16
                    report.append(f"- **Suggested Resolution:** {width}x{height}")

            # 4. Count Tensors
            report.append(f"- **Total Tensors:** {len(reader.tensors)}")
            
            # 5. Author/Source
            author = reader.get_field("general.author")
            if author: report.append(f"- **Author:** {author}")
            
            description = reader.get_field("general.description")
            if description: report.append(f"- **Description:** {description}")

        except ImportError:
            return ("Error: 'gguf' library not found. Please install ComfyUI-GGUF.", 1024, 1024)
        except Exception as e:
            logger.exception("Failed to read GGUF metadata.")
            return (f"Error reading metadata: {str(e)}", 1024, 1024)

        final_report = "
".join(report)
        return (final_report, width, height)

# Registration info managed in __init__.py
