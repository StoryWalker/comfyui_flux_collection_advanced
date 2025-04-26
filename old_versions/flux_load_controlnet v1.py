import folder_paths
import comfy.controlnet



class DiffControlNetLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                              "control_net_name": (folder_paths.get_filename_list("controlnet"), )}}

    RETURN_TYPES = ("CONTROL_NET",)
    FUNCTION = "load_controlnet"

    CATEGORY = "loaders"

    def load_controlnet(self, model, control_net_name):
        controlnet_path = folder_paths.get_full_path_or_raise("controlnet", control_net_name)
        controlnet = comfy.controlnet.load_controlnet(controlnet_path, model)
        return (controlnet,)



import node_helpers
import numpy as np
import torch

from PIL import Image, ImageOps, ImageSequence

    
class LoadImage:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = folder_paths.filter_files_content_types(files, ["image"])
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)
    






##### For preprocess
import os
from typing import Dict

import folder_paths

def load_module(module_path, module_name=None):
    import importlib.util

    if module_name is None:
        module_name = os.path.basename(module_path)
        if os.path.isdir(module_path):
            module_path = os.path.join(module_path, "__init__.py")

    module_spec = importlib.util.spec_from_file_location(module_name, module_path)

    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)

    return module

custom_nodes = folder_paths.get_folder_paths("custom_nodes")
preprocessors_dir_names = ["ControlNetPreprocessors", "comfyui_controlnet_aux"]

preprocessors: list[str] = []
_preprocessors_map = {
    "canny": "CannyEdgePreprocessor",
    "canny_pyra": "PyraCannyPreprocessor",
    "lineart": "LineArtPreprocessor",
    "lineart_anime": "AnimeLineArtPreprocessor",
    "lineart_manga": "Manga2Anime_LineArt_Preprocessor",
    "lineart_any": "AnyLineArtPreprocessor_aux",
    "scribble": "ScribblePreprocessor",
    "scribble_xdog": "Scribble_XDoG_Preprocessor",
    "scribble_pidi": "Scribble_PiDiNet_Preprocessor",
    "scribble_hed": "FakeScribblePreprocessor",
    "hed": "HEDPreprocessor",
    "pidi": "PiDiNetPreprocessor",
    "mlsd": "M-LSDPreprocessor",
    "pose": "DWPreprocessor",
    "openpose": "OpenposePreprocessor",
    "dwpose": "DWPreprocessor",
    "pose_dense": "DensePosePreprocessor",
    "pose_animal": "AnimalPosePreprocessor",
    "normalmap_bae": "BAE-NormalMapPreprocessor",
    "normalmap_dsine": "DSINE-NormalMapPreprocessor",
    "normalmap_midas": "MiDaS-NormalMapPreprocessor",
    "depth": "DepthAnythingV2Preprocessor",
    "depth_anything": "DepthAnythingPreprocessor",
    "depth_anything_v2": "DepthAnythingV2Preprocessor",
    "depth_anything_zoe": "Zoe_DepthAnythingPreprocessor",
    "depth_zoe": "Zoe-DepthMapPreprocessor",
    "depth_midas": "MiDaS-DepthMapPreprocessor",
    "depth_leres": "LeReS-DepthMapPreprocessor",
    "depth_metric3d": "Metric3D-DepthMapPreprocessor",
    "depth_meshgraphormer": "MeshGraphormer-DepthMapPreprocessor",
    "seg_ofcoco": "OneFormer-COCO-SemSegPreprocessor",
    "seg_ofade20k": "OneFormer-ADE20K-SemSegPreprocessor",
    "seg_ufade20k": "UniFormer-SemSegPreprocessor",
    "seg_animeface": "AnimeFace_SemSegPreprocessor",
    "shuffle": "ShufflePreprocessor",
    "teed": "TEEDPreprocessor",
    "color": "ColorPreprocessor",
    "sam": "SAMPreprocessor",
    "tile": "TilePreprocessor"
}


def apply_preprocessor(image, preprocessor, resolution=512):
    raise NotImplementedError("apply_preprocessor is not implemented")


try:
    module_path = None

    for custom_node in custom_nodes:
        custom_node = custom_node if not os.path.islink(custom_node) else os.readlink(custom_node)
        for module_dir in preprocessors_dir_names:
            if module_dir in os.listdir(custom_node):
                module_path = os.path.abspath(os.path.join(custom_node, module_dir))
                break

    if module_path is None:
        raise Exception("Could not find ControlNetPreprocessors nodes")

    module = load_module(module_path)
    print("Loaded ControlNetPreprocessors nodes from", module_path)

    nodes: Dict = getattr(module, "NODE_CLASS_MAPPINGS")
    available_preprocessors: list[str] = getattr(module, "PREPROCESSOR_OPTIONS")

    AIO_Preprocessor = nodes.get("AIO_Preprocessor", None)
    if AIO_Preprocessor is None:
        raise Exception("Could not find AIO_Preprocessor node")

    for name, preprocessor in _preprocessors_map.items():
        if preprocessor in available_preprocessors:
            preprocessors.append(name)

    aio_preprocessor = AIO_Preprocessor()

    def apply_preprocessor(image, preprocessor, resolution=512):
        if preprocessor == "None":
            return image

        if preprocessor not in preprocessors:
            raise Exception(f"Preprocessor {preprocessor} is not implemented")

        preprocessor_cls = _preprocessors_map[preprocessor]
        args = {"preprocessor": preprocessor_cls, "image": image, "resolution": resolution}

        function_name = AIO_Preprocessor.FUNCTION
        res = getattr(aio_preprocessor, function_name)(**args)
        if isinstance(res, dict):
            res = res["result"]

        return res[0]

except Exception as e:
    print(e)

class AV_ControlNetPreprocessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "preprocessor": (["None"] + preprocessors,),
                "sd_version": (["sd15", "sdxl"],),
            },
            "optional": {
                "resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "preprocessor_override": ("STRING", {"default": "None"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("IMAGE", "CNET_NAME")
    FUNCTION = "detect_controlnet"
    CATEGORY = "Art Venture/Loaders"

    def detect_controlnet(self, image, preprocessor, sd_version, resolution=512, preprocessor_override="None"):
        if preprocessor_override != "None":
            if preprocessor_override not in preprocessors:
                print(
                    f"Warning: Not found ControlNet preprocessor {preprocessor_override}. Use {preprocessor} instead."
                )
            else:
                preprocessor = preprocessor_override

        image = apply_preprocessor(image, preprocessor, resolution=resolution)
        control_net_name = detect_controlnet(preprocessor, sd_version)

        return (image, control_net_name)