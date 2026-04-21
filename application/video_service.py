# -*- coding: utf-8 -*-
import logging
import node_helpers
try:
    from domain.models import GenerationSettings, VideoStoryContext
    from infrastructure.image_adapter import TorchImageAdapter
    from infrastructure.sampler_adapter import ComfySamplerAdapter
except ImportError:
    from ..domain.models import GenerationSettings, VideoStoryContext
    from ..infrastructure.image_adapter import TorchImageAdapter
    from ..infrastructure.sampler_adapter import ComfySamplerAdapter

logger = logging.getLogger(__name__)

class VideoGenerationService:
    """
    Application Service:
    Orchestrates the Wan 2.2 video generation process using Domain models 
    and Infrastructure adapters.
    """
    
    def __init__(self, style_map: dict, img_adapter=None, sampler_adapter=None):
        self.style_map = style_map
        self.img_adapter = img_adapter if img_adapter is not None else TorchImageAdapter()
        self.sampler_adapter = sampler_adapter if sampler_adapter is not None else ComfySamplerAdapter()

    def generate_story_segment(self, settings: GenerationSettings, context: VideoStoryContext, 
                               models: dict, reference_image: any) -> tuple:
        """
        Executes a single video segment generation from image and text.
        """
        logger.info(f"[HEX] Starting Application Service: Video Segment Generation")

        # 1. Domain Logic: Combine Text and Styles
        pos_prompt, neg_prompt = context.get_combined_prompts(self.style_map)

        # 2. Infra: Text Conditioning (Standard Comfy Logic)
        tokens_p = models["clip"].tokenize(pos_prompt)
        cond_p, pooled_p = models["clip"].encode_from_tokens(tokens_p, return_pooled=True)
        pos_cond = [[cond_p, {"pooled_output": pooled_p}]]

        tokens_n = models["clip"].tokenize(neg_prompt)
        cond_n, pooled_n = models["clip"].encode_from_tokens(tokens_n, return_pooled=True)
        neg_cond = [[cond_n, {"pooled_output": pooled_n}]]

        # 3. Infra: Image Preparation & Vision Sync
        img_resized = self.img_adapter.resize_for_wan(
            reference_image, settings.width, settings.height, 
            settings.upscale_method, settings.crop_position
        )
        
        cv_out = self.img_adapter.encode_vision(
            models["clip_vision"], 
            self.img_adapter.get_center_crop(img_resized)
        )

        # 4. Infra: Build 5D Context
        device = models["model_high"].load_device
        latent, concat_img, concat_mask = self.img_adapter.build_5d_context(
            models["vae"], settings.num_frames, settings.width, settings.height, img_resized, device
        )

        # 5. Infra: Apply Conditioning Patches
        c_vals = {"clip_vision_output": cv_out, "concat_latent_image": concat_img, "concat_mask": concat_mask}
        p_final = node_helpers.conditioning_set_values(pos_cond, c_vals)
        n_final = node_helpers.conditioning_set_values(neg_cond, c_vals)

        # 6. Infra: Dual Stage Sampling
        latent_dict = {"samples": latent}
        samples = self.sampler_adapter.run_dual_stage_lcm(
            models["model_high"], models["model_low"], settings.seed, 
            settings.steps_high, settings.steps_low, settings.cfg, 
            p_final, n_final, latent_dict, settings.denoise
        )

        # 7. Final Step: Decode and Cleanup
        self.sampler_adapter.final_cleanup()
        video_output = models["vae"].decode(samples["samples"])
        
        # Ensure 4D output [Frames, H, W, C]
        if len(video_output.shape) == 5 and video_output.shape[0] == 1:
            video_output = video_output.squeeze(0)

        # Return (Video Sequence, Final Latent Dictionary)
        return video_output, samples
