# comfyui_flux_collection_advanced/flux_apply_controlnet_preview.py

import os
import random
import json
import numpy as np
import torch
import logging
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
try:
    from comfy import cli_args
except ImportError:
    class MockArgs:
        disable_metadata = False
        disable_save_metadata = False
    cli_args = MockArgs()
    logging.warning("Could not import comfy.cli_args for ApplyControlNetPreview. Metadata saving defaults based on mock.")

# Setup logger for this specific file if needed, or rely on global print/log
DEBUG_PREFIX = "[FluxControlNetApplyPreview PYTHON DEBUG]"
# logging.basicConfig(level=logging.DEBUG) # Uncomment or configure as needed
# logger = logging.getLogger(__name__) # Uncomment or configure as needed

class FluxControlNetApplyPreview:
    """
    Applies ControlNet conditioning (positive only) and displays a preview
    of the input hint image directly on the node.
    Combines functionality of FluxApplyControlNet and FluxPreviewImage.
    """

    # --- Node Definition ---
    RETURN_TYPES = ("CONDITIONING",) # Output is the modified positive conditioning
    RETURN_NAMES = ("positive",)
    FUNCTION = "apply_controlnet_and_preview"
    CATEGORY = "flux_collection_advanced" # Or your desired category
    OUTPUT_NODE = False # Set to False so the CONDITIONING output is easily connectable

    # --- Constants for internal use ---
    _CONTROL_KEY = 'control'
    _APPLY_TO_UNCOND_KEY = 'control_apply_to_uncond'
    _VAE_KEY = 'vae' # Use consistent key if VAE is optional input
    _DEFAULT_PREVIEW_PREFIX = "FluxACNPreview" # Unique prefix for previews from this node
    _RANDOM_CHARS = "abcdefghijklmnopqrstuvwxyz"
    _RANDOM_LENGTH = 5

    @classmethod
    def INPUT_TYPES(cls):
        """ Defines the input types, same as FluxApplyControlNet. """
        return {
            "required": {
                "positive": ("CONDITIONING", ),
                cls._VAE_KEY: ("VAE", ),
                "image": ("IMAGE", {"tooltip": "The hint image to control conditioning and to preview."}), # Updated tooltip
                "control_net": ("CONTROL_NET", ),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
            },
            "optional": {
                #cls._VAE_KEY: ("VAE", ),
            }
        }

    # --- Helper methods (Copied/adapted from FluxApplyControlNet for encapsulation) ---
    # You could also place these in a shared utils file and import them in both classes

    def _validate_apply_inputs(self, positive, control_net, image, strength, start_percent, end_percent, vae):
        """ Validates inputs specific to ControlNet application logic. """
        print(f"{DEBUG_PREFIX} Validating ControlNet Apply inputs...")
        if not isinstance(positive, list): raise TypeError("Input 'positive' must be a list.")
        if not hasattr(control_net, 'copy') or not hasattr(control_net, 'set_cond_hint'): raise TypeError("Input 'control_net' invalid.")
        if not isinstance(image, torch.Tensor): raise TypeError("Input 'image' must be a Tensor.")
        if image.ndim != 4: raise ValueError(f"Input 'image' tensor wrong dimensions: {image.ndim} (expected 4).")
        if image.shape[0] == 0: raise ValueError("Input 'image' tensor is empty.")
        # Add other checks as needed (strength, percent ranges)
        print(f"{DEBUG_PREFIX} ControlNet Apply inputs validated.")
        return True # Indicate success

    def _prepare_control_hint(self, image: torch.Tensor) -> torch.Tensor:
        """ Prepares the control hint image by adjusting dimensions. """
        print(f"{DEBUG_PREFIX} Preparing control hint...")
        if image.ndim != 4: raise ValueError(f"Hint image tensor wrong dimensions: {image.ndim} (expected 4).")
        try:
            control_hint = image.movedim(-1, 1) # BHWC -> BCHW
            print(f"{DEBUG_PREFIX} Control hint prepared. Shape: {control_hint.shape}")
            return control_hint
        except Exception as e:
            print(f"{DEBUG_PREFIX} ERROR: Failed to adjust hint image dimensions: {e}")
            raise RuntimeError(f"Error processing hint image dimensions: {e}") from e

    def _process_single_conditioning(self, conditioning_list: list, base_control_net, control_hint: torch.Tensor, strength: float, timing: tuple[float, float], vae, control_net_cache: dict, extra_concat: list = []) -> list:
        """ Processes a list of conditioning items (positive only here). """
        print(f"{DEBUG_PREFIX} Processing single conditioning list (count: {len(conditioning_list)})...")
        processed_conditioning = []
        for i, conditioning_item in enumerate(conditioning_list):
            print(f"{DEBUG_PREFIX}   Processing item {i}...")
            if not isinstance(conditioning_item, (list, tuple)) or len(conditioning_item) != 2: raise TypeError(f"Conditioning item {i} has invalid structure.")
            if not isinstance(conditioning_item[1], dict): raise TypeError(f"Conditioning item {i} dict missing.")

            tensor_data = conditioning_item[0]
            conditioning_dict_original = conditioning_item[1]
            conditioning_dict_copy = conditioning_dict_original.copy()

            try:
                previous_controlnet_ref = conditioning_dict_copy.get(self._CONTROL_KEY, None)
                if previous_controlnet_ref in control_net_cache:
                    final_control_net = control_net_cache[previous_controlnet_ref]
                    print(f"{DEBUG_PREFIX}     Item {i}: Reusing cached ControlNet.")
                else:
                    print(f"{DEBUG_PREFIX}     Item {i}: Creating new ControlNet instance.")
                    final_control_net = base_control_net.copy()
                    final_control_net = final_control_net.set_cond_hint(control_hint, strength, timing, vae=vae, extra_concat=extra_concat)
                    final_control_net.set_previous_controlnet(previous_controlnet_ref)
                    control_net_cache[previous_controlnet_ref] = final_control_net

                conditioning_dict_copy[self._CONTROL_KEY] = final_control_net
                conditioning_dict_copy[self._APPLY_TO_UNCOND_KEY] = False
                processed_conditioning.append([tensor_data, conditioning_dict_copy])
                print(f"{DEBUG_PREFIX}   Item {i} processed successfully.")

            except AttributeError as e:
                 print(f"{DEBUG_PREFIX} ERROR: Missing attribute/method during ControlNet processing for item {i}: {e}")
                 raise AttributeError(f"Error applying ControlNet (check methods): {e}") from e
            except Exception as e:
                print(f"{DEBUG_PREFIX} ERROR: Unexpected error during processing of item {i}: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"Failed to process conditioning item {i}: {e}") from e
        print(f"{DEBUG_PREFIX} Finished processing single conditioning list.")
        return processed_conditioning

    # --- Main Execution Function ---
    def apply_controlnet_and_preview(self,
                                     positive: list,
                                     control_net,
                                     image: torch.Tensor, # Input image for ControlNet AND Preview
                                     strength: float,
                                     start_percent: float,
                                     end_percent: float,
                                     vae=None,
                                     extra_concat: list = [],
                                     # Include hidden inputs if metadata desired for preview (optional)
                                     prompt=None,
                                     extra_pnginfo=None):
        """ Applies ControlNet and generates preview of the input hint image. """
        print(f"\n{DEBUG_PREFIX} apply_controlnet_and_preview called!")

        # --- 1. Apply ControlNet Logic ---
        processed_positive = [] # Initialize default empty result
        try:
            print(f"{DEBUG_PREFIX} --- Starting ControlNet Application ---")
            self._validate_apply_inputs(positive, control_net, image, strength, start_percent, end_percent, vae)

            if strength == 0:
                print(f"{DEBUG_PREFIX} Strength is 0, skipping ControlNet application.")
                processed_positive = positive # Pass original conditioning through
            else:
                control_hint = self._prepare_control_hint(image)
                control_net_cache = {}
                timing = (start_percent, end_percent)
                processed_positive = self._process_single_conditioning(
                    positive, control_net, control_hint, strength, timing, vae, control_net_cache, extra_concat
                )
            print(f"{DEBUG_PREFIX} --- ControlNet Application Finished ---")

        except Exception as e:
            print(f"{DEBUG_PREFIX} ERROR during ControlNet application phase: {e}")
            import traceback
            traceback.print_exc()
            # Decide how to handle: raise error, return original positive, or return empty?
            # Let's return original positive for now so graph might continue, but log error
            processed_positive = positive # Fallback
            # raise e # Option: re-raise to stop graph

        # --- 2. Generate Preview Logic ---
        preview_results = []
        try:
            print(f"{DEBUG_PREFIX} --- Starting Preview Generation ---")
            # Define preview parameters
            temp_output_dir = folder_paths.get_temp_directory()
            # Create a unique prefix possibly incorporating node ID if available, or random
            prefix_append = "_" + ''.join(random.choice(self._RANDOM_CHARS) for _ in range(self._RANDOM_LENGTH))
            preview_prefix = self._DEFAULT_PREVIEW_PREFIX + prefix_append
            compress_level = 1 # Low compression for previews

            os.makedirs(temp_output_dir, exist_ok=True)

            # Get save path info
            img_h, img_w = image.shape[1:3]
            full_output_folder, filename, counter, subfolder, filename_prefix_resolved = \
                folder_paths.get_save_image_path(preview_prefix, temp_output_dir, img_h, img_w)
            print(f"{DEBUG_PREFIX} Preview Path: Folder='{full_output_folder}', BaseFilename='{filename}', Counter={counter}")

            # Process the input image tensor(s) for preview
            for i, image_tensor_in in enumerate(image):
                print(f"{DEBUG_PREFIX} Previewing image index {i}...")
                try:
                    # Convert tensor to PIL Image
                    img_array = 255. * image_tensor_in.cpu().numpy()
                    img_pil = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
                    print(f"{DEBUG_PREFIX} Preview Image {i} Converted. Size: {img_pil.size}")

                    # Metadata for preview? Usually not needed, keep it simple. Set metadata = None
                    metadata = None
                    # If you *did* want metadata from hidden inputs: copy logic from corrected FluxPreviewImage here.
                    # metadata_disabled = getattr(cli_args, ... ) etc.

                    # Generate filename
                    file = f"{filename}_{counter:05}_.png"
                    full_path = os.path.join(full_output_folder, file)
                    print(f"{DEBUG_PREFIX} Saving Preview Image {i} to {full_path}")

                    # Save the image
                    img_pil.save(full_path, pnginfo=metadata, compress_level=compress_level)
                    print(f"{DEBUG_PREFIX} Preview Image {i} SAVE SUCCESSFUL.")

                    # Add result to list
                    preview_results.append({
                        "filename": file,
                        "subfolder": subfolder,
                        "type": "temp"
                    })
                    print(f"{DEBUG_PREFIX} Appended preview result for image {i}. Current results: {preview_results}")
                    counter += 1

                except Exception as e_inner:
                    print(f"{DEBUG_PREFIX} ERROR processing/saving preview image index {i}: {e_inner}")
                    import traceback
                    traceback.print_exc()
                    continue # Skip this image

            print(f"{DEBUG_PREFIX} Preview generation finished. Final preview_results: {preview_results}")

        except Exception as e_outer:
            print(f"{DEBUG_PREFIX} ERROR during preview generation setup/loop: {e_outer}")
            import traceback
            traceback.print_exc()
            preview_results = [] # Ensure empty list on failure

        # --- 3. Return Combined Results ---
        print(f"{DEBUG_PREFIX} Returning processed_positive (type: {type(processed_positive)}) and UI data (images count: {len(preview_results)}).")
        # Ensure the return format matches node expectations: tuple if multiple outputs defined,
        # single value otherwise. Last item can be the UI dict.
        # Since RETURN_TYPES is ("CONDITIONING",), we return a tuple where the first element
        # is the conditioning, and the second (optional for connection purposes) is the UI dict.
        # However, ComfyUI expects the UI dict as the *only* return value if OUTPUT_NODE=True,
        # or as the *last* element of a tuple if OUTPUT_NODE=False and RETURN_TYPES defined.
        # Let's try returning just the conditioning and the UI dict separately as a tuple.
        # ComfyUI seems to handle `return (data, {"ui": ...})` correctly even if only one RETURN_TYPE is defined.

        return (processed_positive, {"ui": {"images": preview_results}})
        

        ## --- AÑADE ESTA LÍNEA SOLO PARA PROBAR ---
        #print(f"{DEBUG_PREFIX} TEMPORARY TEST: Returning ONLY UI dict.")
        #return {"ui": {"images": preview_results}}
        # -----------------------------------------