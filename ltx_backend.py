# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import sys
import json
import logging
import dataclasses
from collections.abc import Iterator
from typing import cast, Final
import numpy as np
import torch
import safetensors.torch as _sft

logger = logging.getLogger(__name__)

# Intentar importar gguf
_gguf_available: bool
try:
    import gguf
    _gguf_available = True
except ImportError:
    _gguf_available = False


TORCH_COMPATIBLE_QTYPES = (None, 2, 1) # Equivale a None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16

def dequantize_tensor(tensor: object, dtype: torch.dtype | None = None) -> torch.Tensor:
    qtype = getattr(tensor, "tensor_type", None)
    oshape = getattr(tensor, "tensor_shape", None)
    data: torch.Tensor = getattr(tensor, "data", tensor)

    # Si gguf no está disponible, no podemos procesar qtypes avanzados
    if not _gguf_available:
        return data.to(dtype)

    # Mapear qtypes compatibles
    if qtype in (None, gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
        return data.to(dtype)

    if qtype in _DEQUANT_FUNCTIONS:
        if oshape is None:
            raise ValueError(f"tensor_shape required for qtype {qtype}")
        return _dequantize(data, qtype, oshape, dtype=dtype)

    raw = gguf.quants.dequantize(data.cpu().numpy(), qtype)
    return torch.from_numpy(np.array(raw)).to(data.device, dtype=dtype)

def _dequantize(data: torch.Tensor, qtype: gguf.GGMLQuantizationType, oshape: torch.Size, dtype: torch.dtype | None) -> torch.Tensor:
    block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
    rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)
    n_blocks = rows.numel() // type_size
    blocks = rows.reshape((n_blocks, type_size))
    return _DEQUANT_FUNCTIONS[qtype](blocks, block_size, type_size, dtype).reshape(oshape)

def _to_uint32(x: torch.Tensor) -> torch.Tensor:
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)

def _split(blocks: torch.Tensor, *args: int) -> tuple[torch.Tensor, ...]:
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return tuple(torch.split(blocks, dims, dim=1))

def _dequant_BF16(blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype | None) -> torch.Tensor:
    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32).to(dtype)

def _dequant_Q8_0(blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype | None) -> torch.Tensor:
    d, x = _split(blocks, 2)
    return d.view(torch.float16).to(dtype) * x.view(torch.int8)

def _dequant_Q5_0(blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype | None) -> torch.Tensor:
    n = blocks.shape[0]
    d, qh, qs = _split(blocks, 2, 4)
    d = d.view(torch.float16).to(dtype)
    qh = _to_uint32(qh).reshape(n, 1) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape(n, -1, 1, block_size // 2) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape(n, -1)
    return d * ((ql | (qh << 4)).to(torch.int8) - 16)

def _dequant_Q4_0(blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype | None) -> torch.Tensor:
    n = blocks.shape[0]
    d, qs = _split(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    qs = qs.reshape(n, -1, 1, block_size // 2) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    return d * ((qs & 0x0F).reshape(n, -1).to(torch.int8) - 8)

_QK_K = 256
_K_SCALE_SIZE = 12

def _get_scale_min(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = scales.shape[0]
    scales = scales.view(torch.uint8).reshape(n, 3, 4)
    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)
    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    mn = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return sc.reshape(n, 8), mn.reshape(n, 8)

def _dequant_Q5_K(blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype | None) -> torch.Tensor:
    n = blocks.shape[0]
    d, dmin, scales, qh, qs = _split(blocks, 2, 2, _K_SCALE_SIZE, _QK_K // 8)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = _get_scale_min(scales)
    d_sc = (d * sc).reshape(n, -1, 1)
    dm = (dmin * m).reshape(n, -1, 1)
    ql = qs.reshape(n, -1, 1, 32) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qh = qh.reshape(n, -1, 1, 32) >> torch.tensor(list(range(8)), device=d.device, dtype=torch.uint8).reshape(1, 1, 8, 1)
    ql = (ql & 0x0F).reshape(n, -1, 32)
    qh = (qh & 0x01).reshape(n, -1, 32)
    q = ql | (qh << 4)
    return (d_sc * q - dm).reshape(n, _QK_K)

def _dequant_Q4_K(blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype | None) -> torch.Tensor:
    n = blocks.shape[0]
    d, dmin, scales, qs = _split(blocks, 2, 2, _K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = _get_scale_min(scales)
    d_sc = (d * sc).reshape(n, -1, 1)
    dm = (dmin * m).reshape(n, -1, 1)
    qs = qs.reshape(n, -1, 1, 32) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n, -1, 32)
    return (d_sc * qs - dm).reshape(n, _QK_K)

def _dequant_Q6_K(blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype | None) -> torch.Tensor:
    n = blocks.shape[0]
    ql, qh, scales, d = _split(blocks, _QK_K // 2, _QK_K // 4, _QK_K // 16)
    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape(n, _QK_K // 16, 1)
    ql = ql.reshape(n, -1, 1, 64) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    ql = (ql & 0x0F).reshape(n, -1, 32)
    qh = qh.reshape(n, -1, 1, 32) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape(1, 1, 4, 1)
    qh = (qh & 0x03).reshape(n, -1, 32)
    q = (ql | (qh << 4)).to(torch.int8) - 32
    return (d * q.reshape(n, _QK_K // 16, -1)).reshape(n, _QK_K)

def _dequant_Q3_K(blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype | None) -> torch.Tensor:
    n = blocks.shape[0]
    hmask, qs, scales, d = _split(blocks, _QK_K // 8, _QK_K // 4, 12)
    d = d.view(torch.float16).to(dtype)
    lscales = scales[:, :8].reshape(n, 1, 8) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 2, 1)
    lscales = lscales.reshape(n, 16)
    hscales = scales[:, 8:].reshape(n, 1, 4) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape(1, 4, 1)
    hscales = hscales.reshape(n, 16)
    scl = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scl = (scl.to(torch.int8) - 32)
    dl = (d * scl).reshape(n, 16, 1)
    ql = qs.reshape(n, -1, 1, 32) >> torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape(1, 1, 4, 1)
    qh = hmask.reshape(n, -1, 1, 32) >> torch.tensor(list(range(8)), device=d.device, dtype=torch.uint8).reshape(1, 1, 8, 1)
    ql = ql.reshape(n, 16, _QK_K // 16) & 3
    qh = (qh.reshape(n, 16, _QK_K // 16) & 1) ^ 1
    q = ql.to(torch.int8) - (qh << 2).to(torch.int8)
    return (dl * q).reshape(n, _QK_K)

def _dequant_Q2_K(blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype | None) -> torch.Tensor:
    n = blocks.shape[0]
    scales, qs, d, dmin = _split(blocks, _QK_K // 16, _QK_K // 4, 2)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    dl = (d * (scales & 0xF)).reshape(n, _QK_K // 16, 1)
    ml = (dmin * (scales >> 4)).reshape(n, _QK_K // 16, 1)
    shift = torch.tensor([0, 2, 4, 6], device=d.device, dtype=torch.uint8).reshape(1, 1, 4, 1)
    qs = (qs.reshape(n, -1, 1, 32) >> shift) & 3
    qs = qs.reshape(n, _QK_K // 16, 16)
    return (dl * qs - ml).reshape(n, -1)

_KVALUES = torch.tensor([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=torch.int8)

def _dequant_IQ4_NL(blocks: torch.Tensor, block_size: int, type_size: int, dtype: torch.dtype | None) -> torch.Tensor:
    n = blocks.shape[0]
    d, qs = _split(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    qs = qs.reshape(n, -1, 1, block_size // 2) >> torch.tensor([0, 4], device=d.device, dtype=torch.uint8).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape(n, -1, 1).to(torch.int64)
    kv = _KVALUES.to(qs.device).expand(*qs.shape[:-1], 16)
    return d * torch.gather(kv, -1, qs).reshape(n, -1)

if _gguf_available:
    _DEQUANT_FUNCTIONS = {
        gguf.GGMLQuantizationType.BF16: _dequant_BF16,
        gguf.GGMLQuantizationType.Q8_0: _dequant_Q8_0,
        gguf.GGMLQuantizationType.Q5_0: _dequant_Q5_0,
        gguf.GGMLQuantizationType.Q4_0: _dequant_Q4_0,
        gguf.GGMLQuantizationType.Q5_K: _dequant_Q5_K,
        gguf.GGMLQuantizationType.Q4_K: _dequant_Q4_K,
        gguf.GGMLQuantizationType.Q6_K: _dequant_Q6_K,
        gguf.GGMLQuantizationType.Q3_K: _dequant_Q3_K,
        gguf.GGMLQuantizationType.Q2_K: _dequant_Q2_K,
        gguf.GGMLQuantizationType.IQ4_NL: _dequant_IQ4_NL,
    }
else:
    _DEQUANT_FUNCTIONS = {}


class GGUFVirtualBundledLoader:
    def __init__(self, checkpoint_path: str, vae_video_path: str, audio_vae_path: str, connector_path: str) -> None:
        self._checkpoint_path = checkpoint_path
        self._vae_video_path = vae_video_path
        self._audio_vae_path = audio_vae_path
        self._connector_path = connector_path
        self._gguf_reader = None
        self._sf_raw = None

    def _ensure_caches(self) -> None:
        if self._gguf_reader is not None:
            return
        if not _gguf_available:
            raise RuntimeError("El paquete 'gguf' no está instalado. Instálalo en tu entorno virtual.")
        
        import gguf
        import safetensors.torch as _sft

        logger.info(f"GGUF loader: cargando y cacheando {self._checkpoint_path}")
        self._gguf_reader = gguf.GGUFReader(self._checkpoint_path)

        self._sf_raw = {}
        for path in [self._vae_video_path, self._audio_vae_path, self._connector_path]:
            if path and os.path.exists(path):
                self._sf_raw[path] = _sft.load_file(path, device="cpu")
            else:
                self._sf_raw[path] = {}

    def metadata(self, path: str) -> dict[str, object]:
        if not _gguf_available:
            raise RuntimeError("El paquete 'gguf' no está instalado.")
        self._ensure_caches()
        assert self._gguf_reader is not None
        for field in self._gguf_reader.fields.values():
            if field.name == "config":
                parts = field.parts
                raw = bytes(np.array(parts[-1], dtype=np.uint8))
                return json.loads(raw.decode("utf-8").rstrip("\x00"))
        return {}

    def load(self, path: str | list[str], sd_ops: object = None, device: torch.device | None = None) -> object:
        from ltx_core.loader.primitives import StateDict

        if not _gguf_available:
            raise RuntimeError("El paquete 'gguf' no está instalado.")
        self._ensure_caches()

        device = device or torch.device("cpu")
        sd: dict[str, torch.Tensor] = {}

        self._load_gguf_cached(prefix="model.diffusion_model.", sd_ops=sd_ops, device=device, out=sd)
        self._load_safetensors_cached(self._vae_video_path, prefix="vae.", sd_ops=sd_ops, device=device, out=sd)
        self._load_safetensors_cached(self._audio_vae_path, prefix="", sd_ops=sd_ops, device=device, out=sd)
        self._load_safetensors_cached(self._connector_path, prefix="", sd_ops=sd_ops, device=device, out=sd)

        size = sum(t.nbytes for t in sd.values())
        dtype_set = {t.dtype for t in sd.values()}
        return StateDict(sd=sd, device=device, size=size, dtype=dtype_set)

    def _load_gguf_cached(self, prefix: str, sd_ops: object, device: torch.device, out: dict[str, torch.Tensor]) -> None:
        import gguf
        assert self._gguf_reader is not None
        loaded = skipped = 0
        _torch_compat = (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)

        for tensor in self._gguf_reader.tensors:
            raw_name = prefix + tensor.name
            mapped_name = raw_name if sd_ops is None else sd_ops.apply_to_key(raw_name)
            if mapped_name is None:
                skipped += 1
                continue

            arr = np.array(tensor.data)

            if tensor.tensor_type in _torch_compat:
                weight = torch.from_numpy(arr.copy()).to(device=device, dtype=torch.bfloat16)
            else:
                raw_bytes = torch.from_numpy(arr.view(np.uint8).copy())
                raw_bytes.tensor_type = tensor.tensor_type
                raw_bytes.tensor_shape = torch.Size(list(reversed(tensor.shape)))
                weight = dequantize_tensor(raw_bytes, dtype=torch.bfloat16).to(device=device)

            key_value_pairs = [(mapped_name, weight)]
            if sd_ops is not None:
                results = sd_ops.apply_to_key_value(mapped_name, weight)
                key_value_pairs = [(r.new_key, r.new_value) for r in results]

            for key, val in key_value_pairs:
                out[key] = val
            loaded += 1

        logger.info(f"GGUF transformer: cargado {loaded} tensores, omitido {skipped}")

    def _load_safetensors_cached(self, path: str, prefix: str, sd_ops: object, device: torch.device, out: dict[str, torch.Tensor]) -> None:
        if not path or path not in self._sf_raw:
            return
        raw_sd = self._sf_raw[path]
        loaded = skipped = 0

        for name, tensor in raw_sd.items():
            raw_name = prefix + name
            mapped_name = raw_name if sd_ops is None else sd_ops.apply_to_key(raw_name)
            if mapped_name is None:
                skipped += 1
                continue

            t = tensor.to(device=device)
            key_value_pairs = [(mapped_name, t)]
            if sd_ops is not None:
                results = sd_ops.apply_to_key_value(mapped_name, t)
                key_value_pairs = [(r.new_key, r.new_value) for r in results]

            for key, val in key_value_pairs:
                out[key] = val
            loaded += 1

        logger.info(f"Safetensors {os.path.basename(path)}: cargado {loaded} tensores, omitido {skipped}")


def get_device_type(device: str | torch.device | object | None) -> str:
    if device is None:
        return "cpu"
    device_type = getattr(device, "type", None)
    if isinstance(device_type, str):
        return device_type
    if isinstance(device, str):
        try:
            return str(torch.device(device).type)
        except Exception:
            return device
    return "cpu"


def device_supports_fp8(device: str | torch.device | object | None) -> bool:
    return get_device_type(device) == "cuda"


def default_tiling_config():
    from ltx_core.model.video_vae.tiling import TilingConfig, SpatialTilingConfig, TemporalTilingConfig
    return TilingConfig(
        spatial_config=SpatialTilingConfig(tile_size_in_pixels=256, tile_overlap_in_pixels=64),
        temporal_config=TemporalTilingConfig(tile_size_in_frames=32, tile_overlap_in_frames=16),
    )


def extract_video_audio(result: object) -> tuple:
    """
    Extrae video y audio del resultado del pipeline LTX Distilled.
    El pipeline retorna tuple(Iterator[torch.Tensor], Audio) o solo video.

    Returns:
        (video_tensor, audio_waveform_np, sampling_rate)
        audio_waveform_np puede ser None si no hay audio.
    """
    audio_np: np.ndarray | None = None
    sampling_rate: int = 0
    video_tensor: torch.Tensor | None = None

    if isinstance(result, tuple) and len(result) >= 2:
        video_part = result[0]
        audio_part = result[1]
    else:
        video_part = result
        audio_part = None

    # Normalizar video a tensor torch [frames, H, W, 3]
    if isinstance(video_part, torch.Tensor):
        video_tensor = video_part
    elif hasattr(video_part, '__iter__'):
        tensors = list(video_part)
        if tensors and tensors[0].dim() == 3:
            video_tensor = torch.stack(tensors, dim=0)
        else:
            video_tensor = torch.cat(tensors, dim=0)
    else:
        video_tensor = video_part

    # Extraer audio
    if audio_part is not None:
        try:
            waveform = getattr(audio_part, "waveform", None)
            sampling_rate = getattr(audio_part, "sampling_rate", 0)
            if waveform is not None and sampling_rate > 0:
                # waveform: torch.Tensor, posible forma [channels, samples] o [samples]
                w = waveform.detach().cpu()
                if w.dim() == 1:
                    w = w.unsqueeze(0)  # [1, samples]
                elif w.dim() > 2:
                    w = w.reshape(w.shape[-2], w.shape[-1])
                # Normalizar a int16 para WAV
                w_np = w.numpy()
                if w_np.dtype in (np.float32, np.float64):
                    # Asumir rango [-1, 1] o verificar
                    peak = np.abs(w_np).max()
                    if peak > 1.0:
                        w_np = w_np / peak
                    w_np = (w_np * 32767.0).clip(-32768, 32767).astype(np.int16)
                audio_np = w_np
        except Exception as e:
            logger.warning(f"[LTX] No se pudo extraer audio del pipeline: {e}")

    return video_tensor, audio_np, sampling_rate


def save_audio_wav(path: str, waveform_np: np.ndarray, sampling_rate: int) -> None:
    """Guarda audio numpy int16 [channels, samples] como archivo WAV."""
    try:
        from scipy.io import wavfile
        wavfile.write(path, sampling_rate, waveform_np.T if waveform_np.ndim > 1 else waveform_np)
    except Exception as e:
        logger.error(f"[LTX] Error guardando WAV: {e}")
        raise


def mux_video_audio_with_ffmpeg(video_path: str, audio_path: str, output_path: str) -> None:
    """
    Mezcla video MP4 + audio WAV en un solo MP4 usando ffmpeg.
    Requiere ffmpeg instalado en el sistema.
    """
    import subprocess
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"[LTX] Video+audio mezclado exitosamente: {output_path}")
    except subprocess.CalledProcessError as e:
        logger.error(f"[LTX] ffmpeg falló: {e.stderr.decode('utf-8', errors='ignore')}")
        raise RuntimeError(f"ffmpeg no pudo mezclar video+audio: {e}")


class LTXFastVideoPipeline:
    pipeline_kind = "fast"

    @staticmethod
    def create(checkpoint_path: str, gemma_root: str | None, upsampler_path: str, device: torch.device) -> "LTXFastVideoPipeline":
        return LTXFastVideoPipeline(checkpoint_path, gemma_root, upsampler_path, device)

    def __init__(self, checkpoint_path: str, gemma_root: str | None, upsampler_path: str, device: torch.device) -> None:
        from ltx_core.quantization import QuantizationPolicy
        from ltx_pipelines.distilled import DistilledPipeline

        self._checkpoint_path = checkpoint_path
        self._gemma_root = gemma_root
        self._upsampler_path = upsampler_path
        self._device = device
        self._quantization = QuantizationPolicy.fp8_cast() if device_supports_fp8(device) else None

        self.pipeline = DistilledPipeline(
            distilled_checkpoint_path=checkpoint_path,
            gemma_root=cast(str, gemma_root),
            spatial_upsampler_path=upsampler_path,
            loras=[],
            device=device,
            quantization=self._quantization,
        )

    def _run_inference(self, prompt: str, seed: int, height: int, width: int, num_frames: int, frame_rate: float, images: list, tiling_config) -> tuple:
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput
        
        ltx_images = []
        for img in images:
            path = getattr(img, "path", img.get("path") if isinstance(img, dict) else img)
            frame_idx = getattr(img, "frame_idx", img.get("frame_idx") if isinstance(img, dict) else 0)
            strength = getattr(img, "strength", img.get("strength") if isinstance(img, dict) else 1.0)
            ltx_images.append(_LtxImageInput(path, frame_idx, strength))

        return self.pipeline(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=ltx_images,
            tiling_config=tiling_config,
            streaming_prefetch_count=2,
        )


class LTXDistilledGGUFVideoPipeline:
    pipeline_kind = "fast"

    @staticmethod
    def create(checkpoint_path: str, gemma_root: str | None, upsampler_path: str, device: torch.device, *, vae_video_path: str, audio_vae_path: str, connector_path: str) -> "LTXDistilledGGUFVideoPipeline":
        return LTXDistilledGGUFVideoPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            device=device,
            vae_video_path=vae_video_path,
            audio_vae_path=audio_vae_path,
            connector_path=connector_path,
        )

    def __init__(self, checkpoint_path: str, gemma_root: str | None, upsampler_path: str, device: torch.device, *, vae_video_path: str, audio_vae_path: str, connector_path: str) -> None:
        from ltx_core.quantization import QuantizationPolicy
        from ltx_pipelines.distilled import DistilledPipeline

        self._checkpoint_path = checkpoint_path
        self._gemma_root = gemma_root
        self._upsampler_path = upsampler_path
        self._device = device
        self._vae_video_path = vae_video_path
        self._audio_vae_path = audio_vae_path
        self._connector_path = connector_path

        self._quantization = QuantizationPolicy.fp8_cast() if device_supports_fp8(device) else None

        self.pipeline = DistilledPipeline(
            distilled_checkpoint_path=checkpoint_path,
            gemma_root=cast(str, gemma_root),
            spatial_upsampler_path=upsampler_path,
            loras=[],
            device=device,
            quantization=self._quantization,
        )

        self._patch_builders()
        self._install_transformer_cache()

    def _patch_builders(self) -> None:
        loader = GGUFVirtualBundledLoader(
            checkpoint_path=self._checkpoint_path,
            vae_video_path=self._vae_video_path,
            audio_vae_path=self._audio_vae_path,
            connector_path=self._connector_path,
        )

        p = self.pipeline
        p.stage._transformer_builder = dataclasses.replace(p.stage._transformer_builder, model_loader=loader)
        p.image_conditioner._encoder_builder = dataclasses.replace(p.image_conditioner._encoder_builder, model_loader=loader)
        p.upsampler._encoder_builder = dataclasses.replace(p.upsampler._encoder_builder, model_loader=loader)
        p.video_decoder._decoder_builder = dataclasses.replace(p.video_decoder._decoder_builder, model_loader=loader)
        
        if hasattr(p, "audio_decoder") and p.audio_decoder:
            p.audio_decoder._decoder_builder = dataclasses.replace(p.audio_decoder._decoder_builder, model_loader=loader)
            p.audio_decoder._vocoder_builder = dataclasses.replace(p.audio_decoder._vocoder_builder, model_loader=loader)
            
        emb_builder = getattr(p.prompt_encoder, "_embeddings_processor_builder", None)
        if emb_builder is not None:
            p.prompt_encoder._embeddings_processor_builder = dataclasses.replace(emb_builder, model_loader=loader)

    def _install_transformer_cache(self) -> None:
        from ltx_core.layer_streaming import LayerStreamingWrapper
        from ltx_pipelines.utils.helpers import cleanup_memory
        from contextlib import contextmanager

        self._fp8_cache = None
        p = self.pipeline
        stage = p.stage

        if not hasattr(stage, "_transformer_ctx"):
            raise AttributeError("DiffusionStage has no attribute '_transformer_ctx'. GGUF caching is not compatible with this version of ltx-pipelines.")

        def _ensure_cache(**kwargs: object) -> None:
            if self._fp8_cache is not None:
                logger.info("FP8 distilled: reusing CPU-resident cache")
                return
            transformer = stage._build_transformer(device=torch.device("cpu"), **kwargs)
            transformer.eval()
            self._fp8_cache = transformer
            logger.info("FP8 distilled: transformer built and cached in CPU RAM")

        @contextmanager
        def _transformer_ctx(streaming_prefetch_count: int | None, **kwargs: object):
            _ensure_cache(**kwargs)
            assert self._fp8_cache is not None
            prefetch = streaming_prefetch_count if streaming_prefetch_count is not None else 2
            logger.info(f"FP8 distilled: streaming from CPU RAM (prefetch={prefetch})")
            wrapped = LayerStreamingWrapper(
                self._fp8_cache,
                layers_attr="velocity_model.transformer_blocks",
                target_device=stage._device,
                prefetch_count=prefetch,
            )
            try:
                yield wrapped
            finally:
                wrapped.teardown()
                torch.cuda.empty_cache()
                cleanup_memory()

        setattr(stage, "_transformer_ctx", _transformer_ctx)
        logger.debug("GGUF transformer cache installed on DiffusionStage._transformer_ctx")

    def _run_inference(self, prompt: str, seed: int, height: int, width: int, num_frames: int, frame_rate: float, images: list, tiling_config) -> tuple:
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput
        
        ltx_images = []
        for img in images:
            path = getattr(img, "path", img.get("path") if isinstance(img, dict) else img)
            frame_idx = getattr(img, "frame_idx", img.get("frame_idx") if isinstance(img, dict) else 0)
            strength = getattr(img, "strength", img.get("strength") if isinstance(img, dict) else 1.0)
            ltx_images.append(_LtxImageInput(path, frame_idx, strength))

        result = self.pipeline(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=ltx_images,
            tiling_config=tiling_config,
            streaming_prefetch_count=2,
        )
        return result
