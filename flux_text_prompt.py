from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
import node_helpers

class FluxTextPrompt(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        return {
            "required": {
                "text": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."}),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1})
            }
        }
    RETURN_TYPES = (IO.CONDITIONING,)
    OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model.",)
    FUNCTION = "encode"
    
    CATEGORY = "flux_collection_advanced"
    DESCRIPTION = "Encodes a text prompt using a CLIP model into an embedding that can be used to guide the diffusion model towards generating specific images."

    def encode(self, clip, text, guidance):
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")
        tokens = clip.tokenize(text)    

        conditioning = clip.encode_from_tokens_scheduled(tokens)    
    
        c = node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})
        return (c, )