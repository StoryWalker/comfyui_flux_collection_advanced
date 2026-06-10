# -*- coding: utf-8 -*-
"""
[HEX] LTX Native Sampler.
Orquesta el pipeline nativo de ComfyUI para LTX 2.3 con soporte completo
para audio+video y dos pasadas de sampling (estructural + refinamiento).

Usa los nodos nativos de ComfyUI directamente para maxima compatibilidad
con Gemma GGUF y el ecosistema de custom nodes.
"""
import logging

try:
    from infrastructure.error_adapter import hex_error_handler
except ImportError:
    from .infrastructure.error_adapter import hex_error_handler

logger = logging.getLogger(__name__)


class LTXNativeSamplerHex:
    """
    [HEX] LTX Native Video Sampler.

    Pipeline de 2 pasadas:
      1. Estructural a media resolucion (CFG=4, 20 steps)
      2. Refinamiento con upscaling 2x (CFG=1, 4 steps manual sigmas)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "section_sampling": ("STRING", {"default": "SAMPLING PARAMETERS"}),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae_video": ("VAE",),
                "vae_audio": ("VAE",),
                "upscale_model": ("LATENT_UPSCALE_MODEL",),
                "prompt": ("STRING", {"multiline": True, "default": "a slow camera pan across a realistic forest portrait, high definition"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "blurry, low quality, still frame, watermark"}),
                "width": ("INT", {"default": 768, "min": 64, "max": 2048, "step": 32}),
                "height": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 32}),
                "num_frames": ("INT", {"default": 97, "min": 9, "max": 257, "step": 8}),
                "frame_rate": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),

                "section_pass1": ("STRING", {"default": "PASS 1 — STRUCTURAL"}),
                "steps_1": ("INT", {"default": 20, "min": 1, "max": 100}),
                "cfg_1": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name_1": (["euler_ancestral", "euler", "dpmpp_2m", "dpmpp_sde", "heun"], {"default": "euler_ancestral"}),

                "section_pass2": ("STRING", {"default": "PASS 2 — REFINEMENT"}),
                "steps_2": ("INT", {"default": 4, "min": 1, "max": 100}),
                "cfg_2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name_2": (["euler_ancestral", "euler", "dpmpp_2m", "dpmpp_sde", "heun"], {"default": "euler_ancestral"}),
                "manual_sigmas_2": ("STRING", {"default": "0.909375, 0.725, 0.421875, 0.0"}),

                "section_scheduler": ("STRING", {"default": "SCHEDULER"}),
                "max_shift": ("FLOAT", {"default": 2.05, "min": 0.0, "max": 100.0, "step": 0.01}),
                "base_shift": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 100.0, "step": 0.01}),
                "stretch": ("BOOLEAN", {"default": True}),
                "terminal": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.99, "step": 0.01}),

                "section_decode": ("STRING", {"default": "DECODE TILING"}),
                "tile_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 32}),
                "overlap": ("INT", {"default": 64, "min": 0, "max": 4096, "step": 32}),
                "temporal_size": ("INT", {"default": 64, "min": 8, "max": 4096, "step": 4}),
                "temporal_overlap": ("INT", {"default": 8, "min": 4, "max": 4096, "step": 4}),
            },
            "optional": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "execute"
    CATEGORY = "flux_collection_advanced/hex_ltx"

    @hex_error_handler
    def execute(self, **kwargs):
        # ------------------------------------------------------------------
        # Extraer parametros
        # ------------------------------------------------------------------
        model = kwargs["model"]
        clip = kwargs["clip"]
        vae_video = kwargs["vae_video"]
        vae_audio = kwargs["vae_audio"]
        upscale_model = kwargs["upscale_model"]

        prompt = kwargs.get("prompt", "")
        negative_prompt = kwargs.get("negative_prompt", "")
        width = kwargs.get("width", 768)
        height = kwargs.get("height", 512)
        num_frames = kwargs.get("num_frames", 97)
        frame_rate = kwargs.get("frame_rate", 24.0)
        seed = kwargs.get("seed", 0)

        steps_1 = kwargs.get("steps_1", 20)
        cfg_1 = kwargs.get("cfg_1", 4.0)
        sampler_name_1 = kwargs.get("sampler_name_1", "euler_ancestral")

        steps_2 = kwargs.get("steps_2", 4)
        cfg_2 = kwargs.get("cfg_2", 1.0)
        sampler_name_2 = kwargs.get("sampler_name_2", "euler_ancestral")
        manual_sigmas_2 = kwargs.get("manual_sigmas_2", "0.909375, 0.725, 0.421875, 0.0")

        max_shift = kwargs.get("max_shift", 2.05)
        base_shift = kwargs.get("base_shift", 0.95)
        stretch = kwargs.get("stretch", True)
        terminal = kwargs.get("terminal", 0.1)

        tile_size = kwargs.get("tile_size", 512)
        overlap = kwargs.get("overlap", 64)
        temporal_size = kwargs.get("temporal_size", 64)
        temporal_overlap = kwargs.get("temporal_overlap", 8)

        image = kwargs.get("image", None)
        strength = kwargs.get("strength", 1.0)

        logger.info(
            f"[LTX Native Sampler] {width}x{height} @ {frame_rate}fps | "
            f"frames={num_frames} | seed={seed} | Pass1={steps_1}steps CFG={cfg_1} | "
            f"Pass2={steps_2}steps CFG={cfg_2}"
        )

        # ------------------------------------------------------------------
        # 1. Crear latentes vacios
        # ------------------------------------------------------------------
        from comfy_extras.nodes_lt import EmptyLTXVLatentVideo, LTXVConcatAVLatent, LTXVSeparateAVLatent
        from comfy_extras.nodes_lt_audio import LTXVEmptyLatentAudio

        video_latent_out = EmptyLTXVLatentVideo.execute(width=width, height=height, length=num_frames, batch_size=1)
        video_latent = video_latent_out.args[0]

        audio_latent_out = LTXVEmptyLatentAudio.execute(
            frames_number=num_frames, frame_rate=int(frame_rate), batch_size=1, audio_vae=vae_audio
        )
        audio_latent = audio_latent_out.args[0]

        av_latent_out = LTXVConcatAVLatent.execute(video_latent=video_latent, audio_latent=audio_latent)
        av_latent = av_latent_out.args[0]

        # ------------------------------------------------------------------
        # 2. Encode prompts
        # ------------------------------------------------------------------
        from nodes import CLIPTextEncode
        text_encoder = CLIPTextEncode()
        positive = text_encoder.encode(clip, prompt)[0]
        negative = text_encoder.encode(clip, negative_prompt)[0]

        # ------------------------------------------------------------------
        # 3. Aplicar frame_rate al conditioning
        # ------------------------------------------------------------------
        from comfy_extras.nodes_lt import LTXVConditioning
        conditioned = LTXVConditioning.execute(positive=positive, negative=negative, frame_rate=frame_rate)
        positive = conditioned.args[0]
        negative = conditioned.args[1]

        # ------------------------------------------------------------------
        # 4. Configurar ModelSamplingLTXV
        # ------------------------------------------------------------------
        from comfy_extras.nodes_lt import ModelSamplingLTXV
        model_sampled_out = ModelSamplingLTXV.execute(
            model=model, max_shift=max_shift, base_shift=base_shift, latent=av_latent
        )
        model = model_sampled_out.args[0]

        # ------------------------------------------------------------------
        # 5. PASADA 1 — Estructural
        # ------------------------------------------------------------------
        from comfy_extras.nodes_custom_sampler import (
            RandomNoise, KSamplerSelect, CFGGuider, SamplerCustomAdvanced, ManualSigmas
        )
        from comfy_extras.nodes_lt import LTXVScheduler, LTXVCropGuides

        noise_out = RandomNoise.execute(noise_seed=seed)
        noise = noise_out.args[0]

        sampler_out = KSamplerSelect.execute(sampler_name=sampler_name_1)
        sampler = sampler_out.args[0]

        sigmas_out = LTXVScheduler.execute(
            steps=steps_1, max_shift=max_shift, base_shift=base_shift,
            stretch=stretch, terminal=terminal, latent=av_latent
        )
        sigmas = sigmas_out.args[0]

        guider_out = CFGGuider.execute(model=model, positive=positive, negative=negative, cfg=cfg_1)
        guider = guider_out.args[0]

        sampled_1_out = SamplerCustomAdvanced.execute(
            noise=noise, guider=guider, sampler=sampler, sigmas=sigmas, latent_image=av_latent
        )
        output_latent_1 = sampled_1_out.args[0]

        # ------------------------------------------------------------------
        # 6. Separar AV despues de pasada 1
        # ------------------------------------------------------------------
        separated_1 = LTXVSeparateAVLatent.execute(av_latent=output_latent_1)
        video_latent_1 = separated_1.args[0]
        audio_latent_1 = separated_1.args[1]

        # ------------------------------------------------------------------
        # 7. Crop guides (preparar para pasada 2)
        # ------------------------------------------------------------------
        cropped = LTXVCropGuides.execute(positive=positive, negative=negative, latent=video_latent_1)
        positive_crop = cropped.args[0]
        negative_crop = cropped.args[1]
        latent_cropped = cropped.args[2]

        # ------------------------------------------------------------------
        # 8. Upsample latente espacial 2x
        # ------------------------------------------------------------------
        from comfy_extras.nodes_lt_upsampler import LTXVLatentUpsampler
        upsampler_node = LTXVLatentUpsampler()
        upsampled = upsampler_node.upsample_latent(
            samples=latent_cropped, upscale_model=upscale_model, vae=vae_video
        )
        video_latent_upscaled = upsampled[0]

        # ------------------------------------------------------------------
        # 9. Re-concatenar AV para pasada 2
        # ------------------------------------------------------------------
        av_latent_2_out = LTXVConcatAVLatent.execute(
            video_latent=video_latent_upscaled, audio_latent=audio_latent_1
        )
        av_latent_2 = av_latent_2_out.args[0]

        # ------------------------------------------------------------------
        # 10. PASADA 2 — Refinamiento
        # ------------------------------------------------------------------
        noise_2_out = RandomNoise.execute(noise_seed=seed + 1)
        noise_2 = noise_2_out.args[0]

        sampler_2_out = KSamplerSelect.execute(sampler_name=sampler_name_2)
        sampler_2 = sampler_2_out.args[0]

        sigmas_2_out = ManualSigmas.execute(sigmas=manual_sigmas_2)
        sigmas_2 = sigmas_2_out.args[0]

        guider_2_out = CFGGuider.execute(
            model=model, positive=positive_crop, negative=negative_crop, cfg=cfg_2
        )
        guider_2 = guider_2_out.args[0]

        sampled_2_out = SamplerCustomAdvanced.execute(
            noise=noise_2, guider=guider_2, sampler=sampler_2, sigmas=sigmas_2, latent_image=av_latent_2
        )
        output_latent_2 = sampled_2_out.args[0]

        # ------------------------------------------------------------------
        # 11. Separar AV final
        # ------------------------------------------------------------------
        final_separated = LTXVSeparateAVLatent.execute(av_latent=output_latent_2)
        video_latent_final = final_separated.args[0]
        audio_latent_final = final_separated.args[1]

        # ------------------------------------------------------------------
        # 12. Decode video con tiling
        # ------------------------------------------------------------------
        from nodes import VAEDecodeTiled
        decode_node = VAEDecodeTiled()
        images = decode_node.decode(
            vae=vae_video,
            samples=video_latent_final,
            tile_size=tile_size,
            overlap=overlap,
            temporal_size=temporal_size,
            temporal_overlap=temporal_overlap,
        )[0]

        # ------------------------------------------------------------------
        # 13. Decode audio
        # ------------------------------------------------------------------
        from comfy_extras.nodes_lt_audio import LTXVAudioVAEDecode
        audio_out = LTXVAudioVAEDecode.execute(samples=audio_latent_final, audio_vae=vae_audio)
        audio = audio_out.args[0]

        logger.info(f"[LTX Native Sampler] Completado. Video: {images.shape} | Audio: {audio['waveform'].shape}")

        return (images, audio)
