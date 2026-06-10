# Modelos Requeridos para LTX Video 2.3 en ComfyUI

Esta guía detalla los archivos de modelos y carpetas que necesitas colocar en las carpetas nativas de tu instalación de **ComfyUI** para que los nodos `[LTX]` carguen y ejecuten la inferencia de manera correcta.

> **Versión del documento:** 0.4.0-sprint7  
> **Pipeline activo:** `[LTX]` Legacy (optimizado con `torch_compile`, tiling agresivo, streaming prefetch)  
> **Hardware objetivo:** RTX 5080 16GB VRAM (GGUF Q4_K_M obligatorio)

---

## Tabla de Contenidos

1. [Resumen: ¿Qué pipeline usar?](#resumen-qué-pipeline-usar)
2. [Archivos requeridos](#archivos-requeridos)
3. [Nodos disponibles](#nodos-disponibles)
4. [Configuración recomendada por hardware](#configuración-recomendada-por-hardware)
5. [Optimizaciones de performance (v0.4.0+)](#optimizaciones-de-performance-v040)
6. [Diagrama de conexión](#diagrama-de-conexión)
7. [Instrucciones de carga](#instrucciones-de-carga)

---

## Resumen: ¿Qué pipeline usar?

A partir de la versión **0.4.0-sprint7**, el proyecto consolidó todos los pipelines LTX en una única opción:

| Característica | `[HEX] LTX` (Hexagonal) | `[LTX]` Legacy (Wrapper) |
|---|---|---|
| **Formato** | GGUF Q4_K_M + Folder HuggingFace (Gemma) | Igual que HEX |
| **Audio** | ✅ Nativo (2 pasadas AV) | Igual que HEX |
| **Velocidad** | Optimizada (prefetch=4, tiling agresivo, inference_mode) | Igual (delega a HEX) |
| **VRAM mínima** | 12-16 GB | Igual |
| **Nodos** | 3 nodos unificados (Loader, Sampler, Saver) | 4 nodos separados (2 Loaders, Sampler, Saver) |
| **Arquitectura** | Hexagonal pura (Domain → Application → Infrastructure) | Thin wrapper sobre HEX (backward compat) |

> **Recomendación:** Usa `[HEX] LTX` para nuevos workflows. Usa `[LTX]` legacy solo si tienes workflows antiguos que no quieres reconectar.
>
> **Nota histórica:** Los nodos `[HEX] LTX Native` fueron eliminados en Sprint 7. Solo quedan `[HEX] LTX` (hexagonal puro) y `[LTX]` (legacy wrapper).

---

## Archivos requeridos

### Pipeline GGUF con Audio (Recomendado para 16GB VRAM)

Para utilizar el nodo **`[LTX] Distilled GGUF Loader`**, coloca los siguientes 6 archivos/directorios:

| Nombre del Recurso | Archivo/Directorio Recomendado | Ruta en ComfyUI | Descripción |
| :--- | :--- | :--- | :--- |
| **GGUF Checkpoint** | `ltx-2.3-22b-distilled-Q4_K_M.gguf` | `ComfyUI/models/checkpoints/` | Transformador principal de LTX Video 22B cuantizado en GGUF Q4. |
| **Gemma Directory** | `gemma-3-12b-it-qat-q4_0-unquantized/` | `ComfyUI/models/clip/` | Carpeta de HuggingFace que contiene el codificador de texto de Gemma 3. |
| **Spatial Upscaler** | `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` | `ComfyUI/models/upscale_models/` | Modelo de superresolución espacial x2 para escalar el video a resolución final. |
| **VAE Video** | `ltx-2.3-22b-dev_video_vae.safetensors` | `ComfyUI/models/vae/` | Codificador/Decodificador temporal-espacial de video LTX 2.3. |
| **Audio VAE** | `ltx-2.3-22b-dev_audio_vae.safetensors` | `ComfyUI/models/vae/` | Modelo VAE para decodificación y generación de audio sincronizado. |
| **Connector** | `ltx-2.3-22b-dev_embeddings_connectors.safetensors` | `ComfyUI/models/clip/` | Proyector de embeddings que conecta el codificador de texto con el transformador. |

> [!NOTE]
> La carpeta de Gemma debe ser una carpeta completa que incluya los archivos de configuración y tokenización (`tokenizer.model`, `preprocessor_config.json`, `config.json`, `model.safetensors.index.json`, etc.), además de los pesos del modelo de texto.

### Pipeline Fast (Solo para 24GB+ VRAM, sin audio)

| Nombre del Recurso | Archivo/Directorio Recomendado | Ruta en ComfyUI | Descripción |
| :--- | :--- | :--- | :--- |
| **Checkpoint** | `ltx-2.3-22b-distilled.safetensors` | `ComfyUI/models/checkpoints/` | Transformador de LTX en formato nativo SafeTensors (FP8 o BF16). |
| **Gemma Directory** | `gemma-3-12b-it-qat-q4_0-unquantized/` | `ComfyUI/models/clip/` | Carpeta del codificador de texto Gemma 3. |
| **Spatial Upscaler** | `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` | `ComfyUI/models/upscale_models/` | Modelo de superresolución espacial x2. |

> [!WARNING]
> El pipeline **Fast NO genera audio**. El `DistilledPipeline` interno no carga los componentes de audio (`audio_vae`, `connector`) en modo Fast. Si necesitas audio, usa **GGUF**.

---

## Nodos disponibles

| Node | Archivo | Función |
|------|---------|---------|
| **`[HEX] LTX Loader`** | `hex_ltx_loader.py` | Loader unificado (Fast o GGUF en un solo nodo). Arquitectura hexagonal pura. |
| **`[HEX] LTX Sampler`** | `hex_ltx_sampler.py` | Sampler T2V/I2V con audio. Arquitectura hexagonal pura. |
| **`[HEX] LTX Video Saver`** | `hex_ltx_video_saver.py` | Saver MP4 + audio mux. Arquitectura hexagonal pura. |
| **`[LTX] Distilled GGUF Loader`** | `ltx_nodes.py` | Wrapper legacy. Delega a `LTXPipelineAdapter`. Mantiene interfaz estable. |
| **`[LTX] Fast Loader`** | `ltx_nodes.py` | Wrapper legacy. Delega a `LTXPipelineAdapter`. Mantiene interfaz estable. |
| **`[LTX] Video Sampler`** | `ltx_nodes.py` | Wrapper legacy. Delega a `LTXVideoGenerationService`. |
| **`[LTX] Video Saver`** | `ltx_nodes.py` | Wrapper legacy. Delega a `ExportVideoService`. |

---

## Configuración recomendada por hardware

### RTX 5080 / 5070 Ti — 16GB VRAM (Tu caso)

Usa **obligatoriamente** el pipeline **GGUF** con estos parámetros:

**En el Loader:**
- `device`: `"cuda"`
- `gguf_checkpoint`: tu archivo `.gguf`
- `vae_video`, `vae_audio`, `connector`: obligatorios para audio

**En el Sampler:**
- `width`: **640** (o 768 si aceptas más tiempo)
- `height`: **360** (o 512 si aceptas más tiempo)
- `num_frames`: **49** o **65** (97 puede hacer OOM o ser muy lento)
- `frame_rate`: **24**

**Tiempos estimados con optimizaciones aplicadas:**

| Resolución | Frames | Tiempo estimado |
|------------|--------|-----------------|
| 640x360 | 49 | **4-6 minutos** |
| 640x360 | 65 | **6-8 minutos** |
| 768x512 | 49 | **8-10 minutos** |
| 768x512 | 97 | **12-16 minutos** ⚠️ |

### RTX 4090 / 3090 — 24GB VRAM

Puedes usar **Fast** (SafeTensors) para máxima velocidad, o **GGUF** si quieres audio:

| Pipeline | Resolución | Frames | Tiempo estimado |
|----------|------------|--------|-----------------|
| Fast FP8 | 768x512 | 97 | **3-5 minutos** |
| GGUF Q4 | 768x512 | 97 | **6-9 minutos** (con audio) |

---

## Optimizaciones de performance (v0.4.0+)

El backend `ltx_backend.py` incluye las siguientes optimizaciones automáticas:

| Optimización | Qué hace | Ganancia estimada |
|-------------|----------|-------------------|
| **`streaming_prefetch_count=4`** | Precarga 4 capas del transformer en GPU desde el cache CPU. Reduce saltos de memoria. | **10-20%** |
| **Tiling agresivo** (`tile_size=128`, `temporal=16`) | Reduce VRAM usada en VAE decode. Menos `cudaMalloc` overhead. | **15-25%** |
| **`torch.set_float32_matmul_precision('high')`** | Activa TF32 en operaciones matriciales. Aprovechado por RTX 30xx/40xx/50xx. | **5-10%** |
| **`torch.inference_mode()`** | Modo de inferencia optimizado. Desactiva gradientes y bookkeeping. | **5%** |
| **Cache de transformer en CPU RAM** | El modelo GGUF se cachea en RAM del sistema y se hace streaming a GPU por capas. | Evita recarga completa entre ejecuciones. |
| **`LTX_DISABLE_EMPTY_CACHE=1`** | Variable de entorno. Desactiva `torch.cuda.empty_cache()` entre ejecuciones para reducir overhead. | **10-15%** (experimental) |

> **Uso avanzado:** Si tienes suficiente VRAM y no necesitas liberar memoria entre generaciones, ejecuta ComfyUI con `LTX_DISABLE_EMPTY_CACHE=1` para ganar velocidad extra.

**Mejora combinada esperada:** ~30-40% más rápido que la versión sin optimizar.

> **Nota sobre `torch.compile`:** En Linux con Triton instalado, `torch_compile=True` podría dar un 30-50% extra. En Windows no está soportado por PyTorch/Triton, por lo que se omite.

---

## Diagrama de conexión

### GGUF con Audio (16GB VRAM)

```
[LTX Distilled GGUF Loader]
  ├── gguf_checkpoint: "ltx-2.3-22b-distilled-Q4_K_M.gguf"
  ├── gemma_directory: "gemma-3-12b-it-qat-q4_0-unquantized"
  ├── spatial_upscaler: "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
  ├── vae_video: "ltx-2.3-22b-dev_video_vae.safetensors"
  ├── audio_vae: "ltx-2.3-22b-dev_audio_vae.safetensors"
  ├── connector: "ltx-2.3-22b-dev_embeddings_connectors.safetensors"
  └── device: "cuda"
       │
       └──► (LTX_PIPELINE)
              │
              ▼
[LTX Video Sampler]
  ├── ltx_pipeline: ← del loader
  ├── prompt: "your prompt here"
  ├── width: 640
  ├── height: 360
  ├── num_frames: 49
  ├── frame_rate: 24.0
  ├── seed: 42
  └── image: (opcional para I2V)
       │
       └──► (video_tensor, audio_data)
              │
              ▼
[LTX Video Saver]
  ├── images: ← video_tensor del sampler
  ├── audio: ← audio_data del sampler  ←── ¡IMPORTANTE! Conecta este cable
  ├── fps: 24
  └── filename_prefix: "LTX_Video"
       │
       └──► MP4 con audio incluido
```

> ⚠️ **Importante:** El sampler retorna `(video_tensor, audio_data)`. Si no conectas el cable `audio` del sampler al `audio` del saver, el MP4 no tendrá pista de audio.

---

## Instrucciones de carga

1. **Descarga los modelos** según la tabla de archivos requeridos.
2. **Colócalos en las carpetas indicadas** de tu instalación ComfyUI.
3. **Reinicia ComfyUI** para que `folder_paths` escanee las nuevas rutas.
4. Los loaders nativos de LTX escanearán estas carpetas y rellenarán los menús desplegables (`COMBO`) con los nombres de archivo exactos.
5. Si los menús desplegables te aparecen vacíos o con la opción `None`, comprueba que:
   - Las extensiones coincidan exactamente (`.gguf`, `.safetensors` o directorios válidos para Gemma).
   - Los archivos estén en la **carpeta correcta** (no en subcarpetas arbitrarias).

### Plugins requeridos

| Plugin | Versión mínima | Requerido para |
|---|---|---|
| `comfyui-gguf` | ≥ 1.1.4 | Carga de UNET y CLIP en formato GGUF |

---

*Última actualización: 2026-06-10 — Arquitectura hexagonal para LTX + optimizaciones de performance (Sprint 7).*
