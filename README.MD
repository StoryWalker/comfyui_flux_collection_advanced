# Flux Collection Advanced (v0.4.1)

A powerful collection of ComfyUI nodes for **Flux.1** and **Flux.2** architectures, with a fully hexagonal architecture, native NVFP4 support, FLUX Kontext semantic injection, Wan 2.2 video storytelling, LTX Video generation, and global seed control.

---

## Nodes — Flux Image Suite [HEX] & [NVFP4]

| Node | Description |
|------|-------------|
| **[NVFP4] Native Loader Hex** | Native fast loader for E2M1 packaged FLUX models (RTX 50xx Blackwell optimized). |
| **[HEX] Flux GGUF Loader** | Loads full Flux stack: UNET + CLIP (single/dual) + VAE. GGUF and Safetensors. TAESD support. |
| **[HEX] Image Load & Encode** | Image loading with automatic VAE encoding to Latent format. |
| **[HEX] Flux Text Prompt** | Text encoding with up to 4 style presets and configurable guidance scale. |
| **[HEX] Flux Sampler Parameters** | Single-stage sampler with Img2Img (`latent_opt`) and FLUX Kontext/Redux semantic injection (`reference_opt`). |
| **[HEX] Image Comparison** | Side-by-side interactive hover comparison of two images. |
| **[HEX] Global Seed** | Global seed control — synchronizes seed and `control_after_generate` to all Flux Sampler nodes in the workflow without cables. |

## Nodes — Wan 2.2 Video Suite [HEX]

| Node | Description |
|------|-------------|
| **[HEX] Wan Unified Loader** | Loads full Wan 2.2 model stack with GGUF/Safetensors support and optional LoRA. |
| **[HEX] Wan Story Sampler** | Two-stage Hi/Lo sampler with style injection and continuity buffer for video storytelling. |
| **[HEX] Image Loader** | Batch image loader with configurable start index and cap. |
| **[HEX] Loop Storage** | Persists latent frames to disk for cross-queue continuity. |
| **[HEX] Loop Fetcher** | Reads persisted latent frames for chaining video sequences. |
| **[HEX] Prompt Sequencer** | Multi-line prompt sequencer with auto-increment across queue runs. |
| **[HEX] Wan Video Saver** | Saves video with configurable FPS, custom path, and timestamped filenames. |

## Nodes — LTX Video 2.3

| Node | Description |
|------|-------------|
| **[LTX] Distilled GGUF Loader** | Loads LTX 22B GGUF + Gemma folder + VAEs + Upscaler + Connectors. Optimized with `torch_compile` and transformer CPU caching. |
| **[LTX] Fast Loader** | Loads LTX SafeTensors checkpoint + Gemma folder + Upscaler. For 24GB+ VRAM. |
| **[LTX] Video Sampler** | Text-to-Video and Image-to-Video generation with native audio support. 2-pass sampling (structural + refinement). |
| **[LTX] Video Saver** | Saves MP4 with audio muxing via ffmpeg. Interactive video preview via `ltx_video_preview.js`. |

> **Performance note:** As of v0.4.0-sprint7, the `[LTX]` legacy pipeline includes backend optimizations (`torch_compile`, aggressive tiling, streaming prefetch=4, TF32 matmul) that improve generation speed by **50-60%** on RTX 5080 16GB compared to earlier versions.
>
> See [`ltx_models.md`](ltx_models.md) for complete model requirements, connection diagrams, and hardware-specific configurations.



## Installation

1. Install [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) for GGUF support.
2. Clone into `ComfyUI/custom_nodes/`:
   ```
   git clone https://github.com/StoryWalker/comfyui_flux_collection_advanced
   ```
3. Restart ComfyUI.

**Requirements:** `torch`, `numpy`, `Pillow` (included in ComfyUI environment).

---

## Changelog

### v0.4.1-sprint8 — 2026-06-13
- **[NVFP4] Native Loader**: Introduced `NVFP4NativeLoaderHex` with runtime monkey-patching of `comfy.utils.load_torch_file` to support E2M1 packaged models natively on ComfyUI without modifying core files.
- **[Kontext] Semantic Injection**: `[HEX] Flux Sampler Parameters` now supports direct injection of reference latents via `reference_opt` (applies `ReferenceLatent` conditioning) for FLUX Kontext/Redux workflows.
- **[UI] Preview Optimization**: Switched `FluxImagePreviewHex` to JPEG compression (85%) to prevent Event Loop blockages when previewing massive 4K/2K images in the ComfyUI frontend.
- **[CLEANUP]**: Removed all legacy `pending_refactor` directories and untracked old references from git.

### v0.4.0-sprint7 — 2026-06-10
- **[LTX] Performance Optimizations**: Backend `ltx_backend.py` upgraded with `torch_compile`, `streaming_prefetch_count=4`, aggressive tiling, and TF32 matmul precision. **~50-60% speed improvement** on RTX 5080 16GB.
- **[LTX] Audio Fix**: Resolved audio routing UX issue. Audio now correctly muxes into MP4 when connected.
- **[REFACTOR]** Consolidated all LTX pipelines into `[LTX]` legacy nodes. Removed HEX and Native variants to focus development effort.
- **[LTX] Tests**: 15 tests covering pipeline adapter, service, normalization, and legacy nodes.

### v0.3.0 — 2026-04-04
- **[HEX] Global Seed**: New node — global seed control with auto-broadcast to all Flux Sampler nodes. Supports `randomize`, `fixed`, `increment`, `decrement` modes. ON/OFF auto-sync toggle.
- **[HEX] Flux Image Suite**: Full hexagonal migration of Flux GGUF Loader, Text Prompt, Sampler Parameters, and Image Comparison nodes.
- **[FIX]** Clip encoding: `encode_from_tokens_scheduled` → `encode_from_tokens` — resolves meta tensor error with T5-XXL GGUF files.
- **[REFACTOR]** File naming normalization: `hex_flux_*` → `hex_image_*` across all layers.
- **[CLEANUP]** Legacy files moved to `references/deprecated/`.

### v0.2.2 — 2026-03-31
- Sprint 1: Eliminated bare excepts, dynamic dimensions in ContinuityService, conditional DEV/TEST node registration via `COMFYUI_DEV_NODES`.

### v0.2.0 — 2026-02-19
- Flux VRAM Loader Beta, improved GGUF stability, hexagonal architecture foundation.

---

## Architecture

```
comfyui_flux_collection_advanced/
├── domain/            ← Pure domain entities (dataclasses, no side effects)
├── application/       ← Orchestration services
├── infrastructure/    ← External adapters (ComfyUI, torch, I/O)
├── hex_*.py           ← HEX and NVFP4 entry point nodes
└── js/                ← Frontend extensions
```

---

*Developed with TEAM_PRO · Hexagonal Architecture v4.0*
