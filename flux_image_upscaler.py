import torch
import comfy.utils
import folder_paths
from comfy import model_management
from comfy.cli_args import args
from comfy.utils import (  # Reemplaza 'your_project' con el nombre correcto de tu proyecto o módulo
    load_torch_file_safe,
    state_dict_prefix_replace,
    ProgressBar
)
from spandrel import ModelLoader, ImageModelDescriptor


class FluxImageUpscaler:
    """
    A class for upscaling images using various models and methods.

    This class encapsulates the functionality for loading upscale models,
    managing device memory, and performing the image upscaling process.
    It adheres to SOLID principles by having a single responsibility (image upscaling),
    using dependency injection (model loading), and providing a clear interface.

    Attributes:
        upscale_methods (list): A list of supported image upscaling methods.
    """

    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    def __init__(self):
        """
        Initializes the ImageUpscaler class.
        """
        self.model = None # Initialize model attribute.

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input types for the ComfyUI node.

        Returns:
            dict: A dictionary specifying the required and optional input parameters.
        """
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("upscale_models"),),
                "image": ("IMAGE",),
                "upscale_method": (cls.upscale_methods,),
                "scale_by": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale_image"
    CATEGORY = "fluxCollection"

    def upscale_image(self, model_name, image, upscale_method, scale_by):
        """
        Main function to upscale the image.  This method orchestrates the entire
        upscaling process.

        Args:
            model_name (str): The name of the upscale model to use.
            image (torch.Tensor): The input image tensor.
            upscale_method (str): The upscaling method to use.
            scale_by (float): The factor by which to upscale the image.

        Returns:
            torch.Tensor: The upscaled image tensor.

        Raises:
            ValueError: If the model name is invalid or if the image tensor is empty.
            RuntimeError: If there's an unexpected error during model loading or processing.
        """

        if not model_name:
            raise ValueError("Model name cannot be empty.")
        if image.nelement() == 0:
            raise ValueError("Input image tensor is empty.")
        try:
            self._load_model(model_name)
            upscaled_image = self._process_upscaling(image, upscale_method, scale_by)
            return (upscaled_image,)
        except Exception as e:
            raise RuntimeError(f"An error occurred during upscaling: {e}") from e
        finally:
            self._cleanup()  # Ensure model is moved to CPU even if error occurs

    def _load_model(self, model_name):
        """
        Loads the specified upscale model.  Handles loading, potential
        state_dict modifications, and validation.

        Args:
            model_name (str): The name of the model to load.

        Raises:
            ValueError: If the model is not found or is not a valid image model.
            RuntimeError: If there's an error during model loading.
        """

        try:
            model_path = folder_paths.get_full_path("upscale_models", model_name)
            if not model_path:
                raise ValueError(f"Model '{model_name}' not found.")
            sd = load_torch_file_safe(model_path)  # Using the utility function
            #Model loader
            if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
                sd = state_dict_prefix_replace(sd, {"module.":""})  # Using utility
            self.model = ModelLoader().load_from_state_dict(sd).eval()

            if not isinstance(self.model, ImageModelDescriptor): #Replace with the correct model
                raise ValueError("Upscale model must be a single-image model.")

        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found at {model_path}")
        except (OSError, IOError) as e:
            raise RuntimeError(f"Error reading model file: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}") from e

    def _process_upscaling(self, image, upscale_method, scale_by):
        """
        Performs the image upscaling using the loaded model. Manages memory,
        tiling, and final scaling.

        Args:
            image (torch.Tensor): The input image tensor.
            upscale_method (str): The method for upscaling.
            scale_by (float): Scaling factor.

        Returns:
            torch.Tensor: The upscaled image.
        """

        device = model_management.get_torch_device()
        self._manage_memory(image, device)
        self.model.to(device)
        in_img = image.movedim(-1, -3).to(device)

        upscaled_image = self._tiled_upscaling(in_img)

        self.model.to("cpu")  # Move model back to CPU after processing
        upscaled_image = torch.clamp(upscaled_image.movedim(-3, -1), min=0, max=1.0)

        # Apply final scaling using a common upscale method if needed
        return self._final_scale(upscaled_image, upscale_method, scale_by)

    def _manage_memory(self, image, device):
        """
        Manages GPU memory before processing to avoid OOM errors.

        Args:
            image (torch.Tensor): Image tensor for calculating memory requirements.
            device (torch.device): The device where the model will be loaded.
        Raises:
            MemoryError: If there isn't sufficient memory.
        """

        if self.model is None:
            raise ValueError("Model is not loaded.")
        try:
            memory_required = model_management.module_size(self.model.model)
            # Estimate of additional memory used by some models (TODO: Make more accurate)
            memory_required += (512 * 512 * 3) * image.element_size() * max(self.model.scale, 1.0) * 384.0
            memory_required += image.nelement() * image.element_size()
            model_management.free_memory(memory_required, device)  # Using memory management
        except MemoryError as e:
            raise MemoryError(f"Not enough memory to load the model: {e}") from e

    def _tiled_upscaling(self, in_img):
        """
        Performs tiled upscaling to handle large images and prevent OOM errors.

        Args:
            in_img (torch.Tensor): The input image tensor (on the correct device).

        Returns:
            torch.Tensor: The upscaled image tensor.

        Raises:
            RuntimeError: If an OOM error occurs and tile size cannot be reduced further.
        """

        if self.model is None:
            raise ValueError("Model is not loaded.")

        tile = 512
        overlap = 32
        oom = True
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                    in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap
                )
                pbar = ProgressBar(steps)  # Using the ProgressBar class
                s = comfy.utils.tiled_scale(
                    in_img,
                    lambda a: self.model(a),
                    tile_x=tile,
                    tile_y=tile,
                    overlap=overlap,
                    upscale_amount=self.model.scale,
                    pbar=pbar,
                )
                oom = False
            except model_management.OOM_EXCEPTION as e:
                tile //= 2
                if tile < 128:
                    raise RuntimeError("OOM error even with smallest tile size.") from e
        return s

    def _final_scale(self, image, upscale_method, scale_by):
        """
        Applies the final scaling to the image using a common upscale method.

        Args:
            image (torch.Tensor): The upscaled image tensor.
            upscale_method (str): Upscale method for final scaling.
            scale_by (float): The scaling factor.

        Returns:
            torch.Tensor: The final upscaled image tensor.
        """

        samples = image.movedim(-1, 1)
        width = round(samples.shape[3] * scale_by)
        height = round(samples.shape[2] * scale_by)
        s = comfy.utils.common_upscale(samples, width, height, upscale_method, "disabled")
        s = s.movedim(1, -1)
        return s
    
    def _cleanup(self):
        """
        Cleans up resources after processing, moving the model to CPU.
        """
        if self.model is not None:
            self.model.to("cpu")
            #del self.model
            #self.model=None
