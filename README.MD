# Flux Collection Advanced (v0.2.0)

A powerful collection of ComfyUI nodes designed to extract the maximum potential from **Flux.1** and **Flux.2** architectures, with a focus on memory efficiency and GGUF support.

## Key Features

- **Flux VRAM Extreme Loader (BETA):** Specialized loader for low-VRAM GPUs. Features T5 layer truncation and intelligent parameter auditing.
- **Flux GGUF Advanced Loader:** Full GGUF support with automatic 1.15 sampling shift.
- **Adaptive Sampling:** Automatic detection of latent channels and text dimensions.
- **TAESD Integration:** Native support for all TAESD variants for fast previews.

## New in this Version (2026-02-19)

- **[NEW]** `Flux VRAM Loader (BETA)`: Save gigabytes of VRAM by truncating T5 layers.
- **[REFACTOR]** Improved stability in GGUF loading and parameter validation.
- **[DOCS]** Added `team_pro/docs` for technical traceability and configuration guides.

## Installation

1. Ensure you have the [ComfyUI-GGUF](https://github.com/city96/ComfyUI-GGUF) plugin installed for GGUF support.
2. Clone this repo into `custom_nodes`.
3. Requirements: `torch`, `numpy`, `Pillow`.

---
*Developed by TEAM_PRO*
