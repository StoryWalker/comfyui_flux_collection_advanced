# -*- coding: utf-8 -*-
from dataclasses import dataclass, field
from typing import Optional, List

@dataclass(frozen=True)
class GenerationSettings:
    """ Pure Domain Entity for Wan 2.2 Sampling Configuration """
    width: int
    height: int
    num_frames: int
    steps_high: int
    steps_low: int
    cfg: float
    seed: int
    denoise: float
    upscale_method: str = "nearest-exact"
    crop_position: str = "center"

    def __post_init__(self):
        if self.width % 16 != 0 or self.height % 16 != 0:
            raise ValueError("Dimensions must be multiples of 16 for Wan 2.2.")
        if self.num_frames < 1:
            raise ValueError("Video must have at least 1 frame.")

@dataclass
class VideoStoryContext:
    """ Pure Domain Entity for Storytelling state """
    positive_prompt: str
    negative_prompt: str
    style_1: str = "No Style"
    style_2: str = "No Style"
    
    def get_combined_prompts(self, style_map: dict) -> tuple:
        def combine(base, s1, s2, is_pos=True):
            idx = 0 if is_pos else 1
            parts = [base]
            for s in [s1, s2]:
                if s in style_map:
                    style_text = style_map[s][idx]
                    if style_text: parts.append(style_text)
            return ", ".join([p for p in parts if p.strip()])
        
        pos = combine(self.positive_prompt, self.style_1, self.style_2, True)
        neg = combine(self.negative_prompt, self.style_1, self.style_2, False)
        return pos, neg

@dataclass(frozen=True)
class WanModelConfig:
    """ Pure Domain Entity for Wan 2.2 Model Loading Stack """
    model_high_name: str
    model_low_name: str
    clip_name: str
    vae_name: str
    clip_vision_name: str
    sampling_shift: float = 5.0
    weight_dtype: str = "default"
    lora_name: str = "None"
    lora_strength: float = 1.0
    t5_optimization: str = "Layer Truncation"
    t5_layers: int = 16

    def is_gguf(self, name: str) -> bool:
        return name.lower().endswith(".gguf")

@dataclass(frozen=True)
class VideoExportManifest:
    """ Pure Domain Entity for Video Saving details """
    filename_prefix: str
    index: int
    fps: int
    custom_path: str = ""
    timestamp: str = field(default_factory=lambda: "") # To be set by service

@dataclass(frozen=True)
class BufferConfig:
    """ Pure Domain Entity for Continuity Buffer logic """
    mode: str # "Initial Frame" or "Continue from Disk"
    
    @property
    def is_initial(self) -> bool:
        return self.mode in ["Initial Frame", "Reset/Static"]

@dataclass(frozen=True)
class ImageLoadConfig:
    """ Pure Domain Entity for Image Loading configuration """
    image_path: str
    load_cap: int = 0
    start_index: int = 0

# Task-Source: T#12-GGUF
@dataclass(frozen=True)
class FluxModelConfig:
    """ Entidad de dominio pura para la configuracion del stack de carga Flux GGUF """
    unet_name: str
    clip_name1: str
    vae_name: str
    clip_type: str           # "flux" | "flux2" | "sd3" | "sdxl"
    base_type: str           # "flux" | "flux2" | "wan2.1"
    clip_name2: str = "None"
    dequant_dtype: str = "default"
    patch_dtype: str = "default"
    patch_on_device: bool = False

    @property
    def use_single_clip(self) -> bool:
        return self.clip_name2 == "None" or not self.clip_name2

    @property
    def clip_type_normalized(self) -> str:
        """ Normaliza clip_type para la API de ComfyUI """
        return "flux" if self.clip_type in ["flux", "flux2", "ltx"] else self.clip_type

    def is_gguf(self, name: str) -> bool:
        return name.lower().endswith(".gguf")


@dataclass(frozen=True)
class FluxSamplerConfig:
    """ Pure Domain Entity for Flux single-stage sampling """
    width: int
    height: int
    batch_size: int
    seed: int
    steps: int
    cfg: float
    sampler_name: str
    scheduler: str
    denoise: float
    vae_tiling: str = "enabled"  # "enabled" | "disabled"

    def __post_init__(self):
        if self.width % 16 != 0 or self.height % 16 != 0:
            raise ValueError("Width and height must be multiples of 16.")
        if self.steps < 1:
            raise ValueError("Steps must be at least 1.")

@dataclass(frozen=True)
class TextEncodingConfig:
    """ Pure Domain Entity for Flux text prompt encoding """
    text: str
    style1: str
    style2: str
    style3: str
    style4: str
    guidance: float

    def get_styled_prompt(self, style_map: dict) -> str:
        parts = [self.text.strip()]
        for s in [self.style1, self.style2, self.style3, self.style4]:
            if s in style_map:
                pos = style_map[s][0]
                if pos and pos.strip():
                    parts.append(pos.strip())
        return ", ".join(p for p in parts if p)

@dataclass(frozen=True)
class PromptSequence:
    """ Pure Domain Entity for Multi-line Prompt Sequencing """
    raw_text: str
    current_index: int
    auto_increment: bool = True

    def get_current_line(self) -> str:
        lines = [l.strip() for l in self.raw_text.split("\n") if l.strip()]
        if not lines:
            return ""
        # If auto_increment is on, the index provided is usually a fallback 
        # or managed by the service.
        return lines[self.current_index % len(lines)]

    @property
    def next_index(self) -> int:
        return self.current_index + 1


# Task-Source: T#29
@dataclass(frozen=True)
class LTXPipelineConfig:
    """ Entidad de dominio pura para la configuración del pipeline LTX Video 2.3 """
    checkpoint_path: str
    gemma_root: str
    upsampler_path: str
    vae_video_path: str = ""
    audio_vae_path: str = ""
    connector_path: str = ""
    pipeline_type: str = "fast"  # "fast" | "distilled_gguf"
    device: str = "cuda"

    def __post_init__(self):
        if self.pipeline_type not in ("fast", "distilled_gguf"):
            raise ValueError(f"pipeline_type must be 'fast' or 'distilled_gguf', got '{self.pipeline_type}'.")
        if self.device not in ("cuda", "cpu"):
            raise ValueError(f"device must be 'cuda' or 'cpu', got '{self.device}'.")
        if not self.checkpoint_path or self.checkpoint_path == "None":
            raise ValueError("checkpoint_path cannot be empty.")
        if not self.gemma_root or self.gemma_root == "None":
            raise ValueError("gemma_root cannot be empty.")
        if not self.upsampler_path or self.upsampler_path == "None":
            raise ValueError("upsampler_path cannot be empty.")

    @property
    def is_gguf(self) -> bool:
        return self.pipeline_type == "distilled_gguf"


@dataclass(frozen=True)
class LTXGenerationSettings:
    """ Entidad de dominio pura para los parámetros de generación de video LTX 2.3 """
    prompt: str
    width: int
    height: int
    num_frames: int
    frame_rate: float
    seed: int
    strength: float = 1.0
    image_path: str = ""  # Condicionamiento de imagen (I2V) opcional

    def __post_init__(self):
        if self.width % 32 != 0 or self.height % 32 != 0:
            raise ValueError("Width and height must be multiples of 32 for LTX 2.3.")
        if self.num_frames < 9 or self.num_frames > 257:
            raise ValueError("num_frames must be between 9 and 257.")
        # LTX requiere num_frames = 8k + 1 (ej: 9, 17, 25, ..., 97, ..., 257)
        if (self.num_frames - 1) % 8 != 0:
            raise ValueError("num_frames must follow 8k+1 pattern (e.g., 9, 17, 25, ..., 257).")
        if self.frame_rate < 1.0 or self.frame_rate > 120.0:
            raise ValueError("frame_rate must be between 1.0 and 120.0.")
        if self.strength < 0.0 or self.strength > 1.0:
            raise ValueError("strength must be between 0.0 and 1.0.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")

    @property
    def use_image_conditioning(self) -> bool:
        return bool(self.image_path and self.image_path != "None")


