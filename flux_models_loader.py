import comfy.sd
import comfy.utils
import folder_paths
import torch

class FluxModelsLoader:
    """
    A class responsible for loading Flux models, including UNET, CLIP, and VAE components,
    according to the SOLID principles.

    Attributes:
        None

    Methods:
        INPUT_TYPES: Defines the input types for the ComfyUI node.
        load_models: Loads the UNET, CLIP, and VAE models based on the provided parameters.
        _load_unet: Loads the UNET model.
        _load_clip: Loads the CLIP model.
        _load_vae: Loads the VAE model.
        vae_list: Retrieves a list of available VAE names, including approximated ones.
        load_taesd: Loads a TAESD VAE model from approximated components.
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input types for the ComfyUI node.

        Returns:
            dict: A dictionary specifying the required and optional input parameters.
                'required' includes 'unet_name', 'weight_dtype', 'clip_name1', 'clip_name2', 'type', and 'vae_name'.
                'optional' includes 'device' for advanced device configuration.
        """
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],),
                "clip_name1": (folder_paths.get_filename_list("text_encoders"),),
                "clip_name2": (folder_paths.get_filename_list("text_encoders"),),
                "type": (["sdxl", "sd3", "flux", "hunyuan_video"],),
                "vae_name": (cls.vae_list(),),
            },
            "optional": {
                "device": (["default", "cpu"], {"advanced": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE",)
    FUNCTION = "load_models"
    CATEGORY = "flux_collection_advanced"

    def load_models(self, unet_name, weight_dtype, clip_name1, clip_name2, type, device="default", vae_name="ae.safetensors"):
        """
        Loads the UNET, CLIP, and VAE models.

        Args:
            unet_name (str): Name of the UNET model file.
            weight_dtype (str): Data type for model weights (e.g., 'fp8_e4m3fn').
            clip_name1 (str): Name of the first CLIP model file.
            clip_name2 (str): Name of the second CLIP model file.
            type (str): Model type ('sdxl', 'sd3', 'flux', 'hunyuan_video').
            device (str, optional): Device to load the CLIP model on ('default' or 'cpu'). Defaults to "default".
            vae_name (str, optional): Name of the VAE model file. Defaults to "ae.safetensors".

        Returns:
            tuple: A tuple containing the loaded UNET model, CLIP model, and VAE model.

        Raises:
            FileNotFoundError:  If any of the specified model files are not found.
        """
        model = self._load_unet(unet_name, weight_dtype)
        clip = self._load_clip(clip_name1, clip_name2, type, device)
        vae = self._load_vae(vae_name)
        return (model, clip, vae)


    def _load_unet(self, unet_name, weight_dtype):
        """
        Loads the UNET model.  Handles different weight data types.

        Args:
            unet_name (str): The name of the UNET model file.
            weight_dtype (str):  The desired weight data type.  Supports "default", "fp8_e4m3fn", "fp8_e4m3fn_fast", and "fp8_e5m2".

        Returns:
            comfy.sd.ModelPatcher: The loaded UNET model.

        Raises:
            FileNotFoundError: If the UNET model file is not found.
        """
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True  # Enable fast FP8 optimizations
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        return model

    def _load_clip(self, clip_name1, clip_name2, type, device):
        """
        Loads the CLIP model.  Handles different CLIP types and device placement.

        Args:
            clip_name1 (str): The name of the first CLIP model file.
            clip_name2 (str): The name of the second CLIP model file.
            type (str): The type of model, used to determine the CLIP type (e.g., "sdxl", "sd3").
            device (str):  The device to load the CLIP model onto ("default" or "cpu").

        Returns:
            comfy.sd.CLIP: The loaded CLIP model.

        Raises:
            FileNotFoundError: If either of the CLIP model files are not found.
        """
        clip_path1 = folder_paths.get_full_path_or_raise("text_encoders", clip_name1)
        clip_path2 = folder_paths.get_full_path_or_raise("text_encoders", clip_name2)

        if type == "sdxl":
            clip_type = comfy.sd.CLIPType.STABLE_DIFFUSION
        elif type == "sd3":
            clip_type = comfy.sd.CLIPType.SD3
        elif type == "flux":
            clip_type = comfy.sd.CLIPType.FLUX
        elif type == "hunyuan_video":
            clip_type = comfy.sd.CLIPType.HUNYUAN_VIDEO
        else:
            raise ValueError(f"Unsupported CLIP type: {type}")

        model_options = {}
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

        clip = comfy.sd.load_clip(
            ckpt_paths=[clip_path1, clip_path2],
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            clip_type=clip_type,
            model_options=model_options
        )
        return clip

    def _load_vae(self, vae_name):
        """
        Loads the VAE model.  Handles both regular VAEs and TAESD-approximated VAEs.

        Args:
            vae_name (str): The name of the VAE.  Can be a regular VAE filename or "taesd", "taesdxl", "taesd3", "taef1".

        Returns:
            comfy.sd.VAE: The loaded VAE.

        Raises:
            FileNotFoundError: If the VAE model file is not found (for regular VAEs).
        """
        if vae_name in ["taesd", "taesdxl", "taesd3", "taef1"]:
            sd = self.load_taesd(vae_name)
        else:
            vae_path = folder_paths.get_full_path_or_raise("vae", vae_name)
            sd = comfy.utils.load_torch_file(vae_path)
        vae = comfy.sd.VAE(sd=sd)
        return vae

    @staticmethod
    def vae_list():
        """
        Gets a list of available VAE names, including TAESD options.

        Returns:
            list: A list of VAE names.
        """
        vaes = folder_paths.get_filename_list("vae")
        approx_vaes = folder_paths.get_filename_list("vae_approx")
        # Dynamically check for TAESD variants
        taesd_variants = {
            "taesd": ["taesd_encoder.", "taesd_decoder."],
            "taesdxl": ["taesdxl_encoder.", "taesdxl_decoder."],
            "taesd3": ["taesd3_encoder.", "taesd3_decoder."],
            "taef1": ["taef1_encoder.", "taef1_decoder."],
        }

        for variant, prefixes in taesd_variants.items():
            if all(any(v.startswith(prefix) for v in approx_vaes) for prefix in prefixes):
                vaes.append(variant)
        return vaes

    @staticmethod
    def load_taesd(name):
        """
        Loads a TAESD VAE model from its approximated encoder and decoder components.

        Args:
            name (str):  The name of the TAESD variant ("taesd", "taesdxl", "taesd3","taef1").

        Returns:
            dict: The loaded TAESD state dictionary.

        Raises:
            FileNotFoundError: If the TAESD encoder or decoder files are not found.
            StopIteration: If expected encoder/decoder files are missing in the directory.
        """
        sd = {}
        approx_vaes = folder_paths.get_filename_list("vae_approx")

        try:
            encoder = next(filter(lambda a: a.startswith(f"{name}_encoder."), approx_vaes))
            decoder = next(filter(lambda a: a.startswith(f"{name}_decoder."), approx_vaes))
        except StopIteration:
             raise FileNotFoundError(f"Missing encoder or decoder for TAESD variant: {name}")
        
        enc = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", encoder))
        for k in enc:
            sd[f"taesd_encoder.{k}"] = enc[k]

        dec = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae_approx", decoder))
        for k in dec:
            sd[f"taesd_decoder.{k}"] = dec[k]

        # Set scaling factors based on the TAESD variant
        if name == "taesd":
            sd["vae_scale"] = torch.tensor(0.18215)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesdxl":
            sd["vae_scale"] = torch.tensor(0.13025)
            sd["vae_shift"] = torch.tensor(0.0)
        elif name == "taesd3":
            sd["vae_scale"] = torch.tensor(1.5305)
            sd["vae_shift"] = torch.tensor(0.0609)
        elif name == "taef1":
            sd["vae_scale"] = torch.tensor(0.3611)
            sd["vae_shift"] = torch.tensor(0.1159)
        return sd