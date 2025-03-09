from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
import node_helpers

class FluxTextPrompt(ComfyNodeABC):
    """
    A ComfyUI node that encodes a text prompt using a CLIP model into an embedding
    that can be used to guide a diffusion model.
    """

    @classmethod
    def INPUT_TYPES(s) -> InputTypeDict:
        """
        Defines the required input types for the node.

        Returns:
            dict: A dictionary describing the required input types.
        """
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
    DESCRIPTION = "Encodes a text prompt using a CLIP model into an embedding to guide the diffusion model."

    def encode(self, clip, text, guidance):
        """
        Encodes a text prompt using a CLIP model.

        Args:
            clip: The CLIP model for encoding.
            text: The text to be encoded.
            guidance: The guidance value for the conditioning.

        Returns:
            tuple: A tuple containing the resulting conditioning.
        """
        try:
            self._validate_clip(clip)
            tokens = self._tokenize_text(clip, text)
            conditioning = self._encode_tokens(clip, tokens)
            conditioned = self._apply_guidance(conditioning, guidance)
            return (conditioned,)
        except Exception as e:
            print(f"Error in FluxTextPrompt: {e}")
            raise

    def _validate_clip(self, clip):
        """
        Validates that the CLIP model is not None.

        Args:
            clip: The CLIP model to validate.

        Raises:
            RuntimeError: If the CLIP model is None.
        """
        if clip is None:
            raise RuntimeError("ERROR: clip input is invalid: None\n\nIf the clip is from a checkpoint loader node your checkpoint does not contain a valid clip or text encoder model.")

    def _tokenize_text(self, clip, text):
        """
        Tokenizes the text using the CLIP model.

        Args:
            clip: The CLIP model for tokenization.
            text: The text to be tokenized.

        Returns:
            The resulting tokens.
        """
        try:
            return clip.tokenize(text)
        except Exception as e:
            raise RuntimeError(f"Error during tokenization: {e}")

    def _encode_tokens(self, clip, tokens):
        """
        Encodes the tokens using the CLIP model.

        Args:
            clip: The CLIP model for encoding.
            tokens: The tokens to be encoded.

        Returns:
            The resulting conditioning from encoding.
        """
        try:
            return clip.encode_from_tokens_scheduled(tokens)
        except Exception as e:
            raise RuntimeError(f"Error during token encoding: {e}")

    def _apply_guidance(self, conditioning, guidance):
        """
        Applies the guidance value to the conditioning.

        Args:
            conditioning: The conditioning to apply guidance to.
            guidance: The guidance value.

        Returns:
            The conditioning with the applied guidance.
        """
        try:
            return node_helpers.conditioning_set_values(conditioning, {"guidance": guidance})
        except Exception as e:
            raise RuntimeError(f"Error applying guidance: {e}")