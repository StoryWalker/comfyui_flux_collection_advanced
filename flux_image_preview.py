import os
import random
import json
import numpy as np
import torch
import logging
from PIL import Image
from PIL.PngImagePlugin import PngInfo # Import PngInfo from its specific submodule

# Assuming folder_paths and comfy.cli_args are available in the environment
# If comfy.cli_args is not reliably available, default 'disable_metadata' to False
import folder_paths
try:
    # Attempt to import the official cli_args
    from comfy import cli_args
    print("[FluxImagePreview] Successfully imported comfy.cli_args")
except ImportError:
    # Create a mock args object if comfy.cli_args is unavailable
    print("[FluxImagePreview] WARNING: Could not import comfy.cli_args. Using mock object.")
    class MockArgs:
        # Add attributes that might be checked, defaulting to safe values
        disable_metadata = False
        disable_save_metadata = False # Add common alternative name just in case
    cli_args = MockArgs()
    # No need for logging warning here as we print directly


# Using print for debug as it's often easier to spot in the main ComfyUI console
DEBUG_PREFIX = "[FluxImagePreview PYTHON DEBUG]"

class FluxImagePreview:
    """
    Generates temporary preview images and displays them in the UI.

    This node saves input images to a temporary directory specified by ComfyUI's
    folder_paths settings. It does not take a filename prefix as input,
    instead using a default prefix combined with a random string to ensure
    temporary files are distinct. It's designed as an output node primarily
    for UI preview purposes. Metadata (prompt, extra_pnginfo) can optionally
    be embedded if not disabled globally.
    """
    # This indicates the node doesn't pass data to subsequent nodes via return values
    RETURN_TYPES = ()
    # The function name within the class that will be executed
    FUNCTION = "preview_images"
    # This node is an endpoint in the graph execution
    OUTPUT_NODE = True
    # Category for the node in the ComfyUI menu
    CATEGORY = "flux_collection_advanced"

    # Constants for internal use
    _DEFAULT_PREFIX = "FluxPreview"
    _RANDOM_CHARS = "abcdefghijklmnopqrstuvwxyz"
    _RANDOM_LENGTH = 5

    def __init__(self):
        """
        Initializes the FluxImagePreview node, setting up temporary storage parameters.
        """
        self.output_dir = "" # Initialize to avoid potential issues if try fails early
        try:
            self.output_dir = folder_paths.get_temp_directory()
            if not self.output_dir:
                self.output_dir = os.path.join(folder_paths.get_output_directory(), "temp")
                print(f"{DEBUG_PREFIX} WARNING: Could not retrieve specific temp directory, using fallback: {self.output_dir}")
            os.makedirs(self.output_dir, exist_ok=True) # Ensure the temp directory exists
        except Exception as e:
            print(f"{DEBUG_PREFIX} ERROR: Failed to get or create temporary directory: {e}")
            self.output_dir = os.path.join(folder_paths.get_output_directory(), "temp_fallback")
            print(f"{DEBUG_PREFIX} WARNING: Using fallback temporary directory due to error: {self.output_dir}")
            os.makedirs(self.output_dir, exist_ok=True)

        # Type identifier for the output result dictionary
        self.type = "temp"
        # Generate a random suffix for temporary filenames to avoid collisions
        self.prefix_append = "_temp_" + ''.join(random.choice(self._RANDOM_CHARS) for _ in range(self._RANDOM_LENGTH))
        # Use lower compression for faster temporary saving
        self.compress_level = 1
        # Use print instead of logger for easier spotting in console
        print(f"{DEBUG_PREFIX} Initialized. Temp Dir: '{self.output_dir}', Prefix Append: '{self.prefix_append}'")

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input types accepted by the node.
        """
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "The images to preview."}),
            },
            "hidden": {
                # Hidden inputs automatically connected if available in the workflow
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    def _validate_inputs(self, images):
        """ Basic validation for the images input. """
        print(f"{DEBUG_PREFIX} Validating inputs...")
        if images is None:
            print(f"{DEBUG_PREFIX} ERROR: Input 'images' is None.")
            raise ValueError("Input 'images' cannot be None.")
        if not isinstance(images, torch.Tensor):
            print(f"{DEBUG_PREFIX} ERROR: Input 'images' is not a Tensor, type is {type(images)}.")
            raise TypeError(f"Input 'images' must be a torch.Tensor, but got {type(images)}.")
        if images.ndim != 4:
            print(f"{DEBUG_PREFIX} ERROR: Input 'images' has incorrect dimensions: {images.ndim} (expected 4).")
            raise ValueError(f"Expected 'images' tensor to have 4 dimensions (Batch, Height, Width, Channels), but got {images.ndim}.")
        if images.shape[0] == 0:
            print(f"{DEBUG_PREFIX} ERROR: Input 'images' tensor is empty (batch size is 0).")
            raise ValueError("Input 'images' tensor cannot be empty (batch size is 0).")
        print(f"{DEBUG_PREFIX} Input images validated. Shape: {images.shape}, Type: {images.dtype}")

    def preview_images(self, images: torch.Tensor, prompt=None, extra_pnginfo=None):
        """
        Saves the input images to the temporary directory and returns metadata for UI preview.
        """
        # --- DEBUG PRINT ---
        print(f"\n{DEBUG_PREFIX} preview_images method called!")
        print(f"{DEBUG_PREFIX} Received prompt type: {type(prompt)}")
        print(f"{DEBUG_PREFIX} Received extra_pnginfo type: {type(extra_pnginfo)}")

        try:
            self._validate_inputs(images)
            # --- DEBUG PRINT ---
            print(f"{DEBUG_PREFIX} Input validation successful.")
        except (ValueError, TypeError) as e:
            # Error already printed in _validate_inputs
            print(f"{DEBUG_PREFIX} Raising validation exception: {e}")
            raise e # Re-raise to stop execution and report error in ComfyUI

        # --- Prepare Filename Prefix and Path ---
        filename_prefix = self._DEFAULT_PREFIX + self.prefix_append
        # --- DEBUG PRINT ---
        print(f"{DEBUG_PREFIX} Generated filename_prefix: {filename_prefix}")
        print(f"{DEBUG_PREFIX} Using output directory: {self.output_dir}")
        try:
            # Correctly get Height and Width from tensor shape (index 1 and 2)
            image_height, image_width = images.shape[1:3]
            full_output_folder, filename, counter, subfolder, filename_prefix_resolved = \
                folder_paths.get_save_image_path(filename_prefix, self.output_dir, image_height, image_width)
            # --- DEBUG PRINT ---
            print(f"{DEBUG_PREFIX} Save path resolved: Folder='{full_output_folder}', BaseFilename='{filename}', Counter={counter}, Subfolder='{subfolder}'")
        except Exception as e:
            print(f"{DEBUG_PREFIX} ERROR: Failed to determine save image path using prefix '{filename_prefix}': {e}")
            import traceback
            traceback.print_exc() # Print full traceback
            raise RuntimeError(f"Error determining save path: {e}") from e

        # --- Process and Save Images ---
        results = []
        # --- DEBUG PRINT ---
        print(f"{DEBUG_PREFIX} Starting image processing loop for {images.shape[0]} image(s)...")
        for i, image_tensor in enumerate(images):
            # --- DEBUG PRINT ---
            print(f"{DEBUG_PREFIX} --- Processing image index {i} ---")
            batch_number = i

            try:
                # --- DEBUG PRINT ---
                print(f"{DEBUG_PREFIX} Image {i}: Converting tensor to NumPy array...")
                img_array = 255. * image_tensor.cpu().numpy()
                print(f"{DEBUG_PREFIX} Image {i}: Clipping and converting to uint8...")
                img_array_clipped = np.clip(img_array, 0, 255).astype(np.uint8)
                print(f"{DEBUG_PREFIX} Image {i}: Creating PIL Image from array...")
                img_pil = Image.fromarray(img_array_clipped)
                # --- DEBUG PRINT ---
                print(f"{DEBUG_PREFIX} Image {i}: Conversion to PIL successful. Size: {img_pil.size}")

                # --- !!! START OF CORRECTED METADATA HANDLING !!! ---
                metadata = None
                # Safely check for common metadata disabling attributes, default to False (metadata enabled)
                # This prevents AttributeError if the attribute doesn't exist in cli_args or the MockArgs
                metadata_disabled = getattr(cli_args, 'disable_metadata', False) or \
                                    getattr(cli_args, 'disable_save_metadata', False) # Common alternative name

                # --- DEBUG PRINT ---
                print(f"{DEBUG_PREFIX} Image {i}: Checking metadata status. Flag 'disable_metadata'/'disable_save_metadata' value is: {metadata_disabled}")

                if not metadata_disabled:
                    print(f"{DEBUG_PREFIX} Image {i}: Metadata is enabled. Creating PngInfo...")
                    metadata = PngInfo() # Create metadata object only if enabled
                    if prompt is not None:
                        try:
                            metadata.add_text("prompt", json.dumps(prompt))
                            print(f"{DEBUG_PREFIX} Image {i}: Added 'prompt' metadata.")
                        except TypeError as json_e:
                             print(f"{DEBUG_PREFIX} WARNING: Could not serialize prompt to JSON for metadata: {json_e}")
                    if extra_pnginfo is not None and isinstance(extra_pnginfo, dict):
                        print(f"{DEBUG_PREFIX} Image {i}: Processing extra_pnginfo dict...")
                        for key, value in extra_pnginfo.items():
                             try:
                                 # Limit metadata value size to prevent errors with huge data
                                 json_value_str = json.dumps(value)
                                 max_len = 10000 # Limit metadata values to 10k chars, adjust if needed
                                 if len(json_value_str) > max_len:
                                      print(f"{DEBUG_PREFIX} WARNING: Truncating metadata for key '{key}' due to excessive length ({len(json_value_str)} chars).")
                                      json_value_str = json_value_str[:max_len] + "...(truncated)"
                                 metadata.add_text(str(key), json_value_str)
                                 # Print only the key for brevity unless debugging metadata specifically
                                 # print(f"{DEBUG_PREFIX} Image {i}: Added extra metadata '{key}': {json_value_str[:100]}...")
                             except TypeError as json_e:
                                 print(f"{DEBUG_PREFIX} WARNING: Could not serialize extra_pnginfo item '{key}' to JSON: {json_e}")
                    elif extra_pnginfo is not None:
                         print(f"{DEBUG_PREFIX} WARNING: extra_pnginfo is not a dictionary, skipping. Type: {type(extra_pnginfo)}")
                else:
                     print(f"{DEBUG_PREFIX} Image {i}: Metadata saving is disabled via cli_args.")
                # --- !!! END OF CORRECTED METADATA HANDLING !!! ---


                # --- Filename Generation and Saving ---
                filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
                file = f"{filename_with_batch_num}_{counter:05}_.png" # Using PNG format
                full_path = os.path.join(full_output_folder, file)
                # --- DEBUG PRINT ---
                print(f"{DEBUG_PREFIX} Image {i}: Attempting to save to: {full_path} with compress_level={self.compress_level}")

                # Ensure directory exists right before saving
                os.makedirs(full_output_folder, exist_ok=True)

                img_pil.save(full_path, pnginfo=metadata, compress_level=self.compress_level)
                # --- DEBUG PRINT ---
                print(f"{DEBUG_PREFIX} Image {i}: SAVE SUCCESSFUL to {full_path}")

                # --- Prepare result dictionary ---
                result_data = {
                    "filename": file,         # Filename relative to the type directory
                    "subfolder": subfolder,   # Subfolder relative to the type directory
                    "type": self.type         # Should be "temp"
                }
                # --- DEBUG PRINT ---
                # print(f"{DEBUG_PREFIX} Image {i}: PRE-APPEND Results list: {results}") # Can be verbose
                print(f"{DEBUG_PREFIX} Image {i}: Appending result data: {result_data}")
                results.append(result_data)
                # --- DEBUG PRINT ---
                print(f"{DEBUG_PREFIX} Image {i}: POST-APPEND Results list: {results}")

                counter += 1
                print(f"{DEBUG_PREFIX} Image {i}: Processing finished. Counter incremented to {counter}.")


            except Exception as e:
                # Log error for the specific image but continue processing others
                print(f"{DEBUG_PREFIX} ERROR: Failed during processing or saving for image index {batch_number}: {e}")
                import traceback
                traceback.print_exc() # Print full traceback to console
                print(f"{DEBUG_PREFIX} WARNING: Continuing to next image despite error on index {batch_number}.")
                continue # Skip appending this image and continue loop

        # --- Return the final results ---
        print(f"{DEBUG_PREFIX} Image processing loop finished.")
        print(f"{DEBUG_PREFIX} FINAL Results list before return: {results}")
        print(f"{DEBUG_PREFIX} Returning dictionary: {{'ui': {{'images': results}}}}")
        return {"ui": {"images": results}}