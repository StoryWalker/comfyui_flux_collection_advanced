# /application/domain/entities.py
__version__ = "1.4.2"
# 1.4.2 - Added full_path to ImageSaveResult.
# 1.4.1 - Added subfolder option to ImageSaveConfig.
# 1.4.0 - Added entities for image saving process.
# 1.3.0 - Added Image entity for VAE decoding.
# 1.2.0 - Added entities for latent generation and sampling.
# 1.1.0 - Added entities for text prompt encoding.
# 1.0.0 - Initial entities for model loading.

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

# --- Type definitions ---
WeightDType = Literal["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"]
ClipArchitectureType = Literal["flux", "sd3", "sdxl", "hunyuan_video"]
DeviceType = Literal["default", "cpu"]

# --- Data classes ---
@dataclass(frozen=True)
class UnetConfig:
    model_name: str
    weight_dtype: WeightDType

@dataclass(frozen=True)
class ClipConfig:
    clip_name1: str
    clip_name2: str
    architecture_type: ClipArchitectureType
    device: DeviceType

@dataclass(frozen=True)
class VaeConfig:
    vae_name: str

@dataclass(frozen=True)
class LoadedModels:
    unet: Any
    clip: Any
    vae: Any

@dataclass(frozen=True)
class Style:
    name: str
    positive_prompt: str
    negative_prompt: str

@dataclass(frozen=True)
class Conditioning:
    data: Any

@dataclass(frozen=True)
class Latent:
    samples: Any

@dataclass(frozen=True)
class Image:
    data: Any

# --- ESTA ES LA CLASE QUE ESTÁ CAUSANDO EL ERROR DE IMPORTACIÓN ---
@dataclass(frozen=True)
class PromptConfig:
    clip_model: Any
    base_text: str
    style_names: List[str] = field(default_factory=list)
    guidance: float = 3.5
# ----------------------------------------------------------------

@dataclass(frozen=True)
class SamplerConfig:
    model: Any
    seed: int
    steps: int
    cfg: float
    sampler_name: str
    scheduler: str
    positive_conditioning: Conditioning
    denoise: float = 1.0

# --- NEW ENTITIES FOR IMAGE SAVING ---
@dataclass(frozen=True)
class ImageMetadata:
    """Holds metadata to be embedded in the saved image."""
    prompt: Optional[Dict[str, Any]] = None
    extra_info: Optional[Dict[str, Any]] = None

@dataclass(frozen=True)
class ImageSaveConfig:
    """Configuration for the image saving operation."""
    filename_prefix: str
    metadata: ImageMetadata
    subfolder: Optional[str] = None # <-- CAMBIO: Añadido campo para la subcarpeta
    compress_level: int = 4

@dataclass(frozen=True)
class ImageSaveResult:
    """Holds the result of a save operation for UI feedback."""
    filename: str
    subfolder: str
    full_path: str  # <-- CAMBIO: Añadido para la ruta completa
    type: str = "output"