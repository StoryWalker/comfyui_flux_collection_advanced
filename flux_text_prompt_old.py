from comfy.comfy_types import IO, ComfyNodeABC, InputTypeDict
import node_helpers
import os
import re
import folder_paths

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

        s.styles_csv = s._load_styles_csv(os.path.join(folder_paths.base_path, "styles.csv"))
        return {
            "required": {
                "text": (IO.STRING, {"multiline": True, "dynamicPrompts": True, "tooltip": "The text to be encoded."}),
                "clip": (IO.CLIP, {"tooltip": "The CLIP model used for encoding the text."}),
                "style1": (list(s.styles_csv.keys()),),
                "style2": (list(s.styles_csv.keys()),),
                "style3": (list(s.styles_csv.keys()),),
                "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1,"tooltip": "The guidance value for the conditioning."})
            }
        }
    RETURN_TYPES = (IO.CONDITIONING,)
    OUTPUT_TOOLTIPS = ("A conditioning containing the embedded text used to guide the diffusion model.",)
    FUNCTION = "encode"

    CATEGORY = "flux_collection_advanced"
    DESCRIPTION = "Encodes a text prompt using a CLIP model into an embedding to guide the flux model."

    def encode(self, clip, text, style1, style2, style3, guidance):
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
            style1 = self.styles_csv[style1][0]
            style2 = self.styles_csv[style2][0]
            style3 = self.styles_csv[style3][0]
            text = text + ', ' + style1 + ', ' + style2 + ', ' + style3
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
        
    def _load_styles_csv(styles_path: str):
        """Loads csv file with styles. It has only one column.
        Ignore the first row (header).
        positive_prompt are strings separated by comma. Each string is a prompt.
        negative_prompt are strings separated by comma. Each string is a prompt.

        Returns:
            list: List of styles. Each style is a dict with keys: style_name and value: [positive_prompt, negative_prompt]
        """
        styles = {"Error loading styles.csv, check the console": ["",""]}
        if not os.path.exists(styles_path):
            print(f"""Error. No styles.csv found. Put your styles.csv in the root directory of ComfyUI. Then press "Refresh".
                  Your current root directory is: {folder_paths.base_path}
            """)
            return styles
        try:
            with open(styles_path, "r", encoding="utf-8") as f:    
                styles = [[x.replace('"', '').replace('\n','') for x in re.split(',(?=(?:[^"]*"[^"]*")*[^"]*$)', line)] for line in f.readlines()[1:]]
                styles = {x[0]: [x[1],x[2]] for x in styles}
        except Exception as e:
            print(f"""Error loading styles.csv. Make sure it is in the root directory of ComfyUI. Then press "Refresh".
                    Your current root directory is: {folder_paths.base_path}
                    Error: {e}
            """)
        return styles