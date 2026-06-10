# Modelos Requeridos para LTX Video 2.3 en ComfyUI

Esta guía detalla los archivos de modelos y carpetas que necesitas colocar en las carpetas nativas de tu instalación de **ComfyUI** para que los nodos `[LTX]` carguen y ejecuten la inferencia de manera correcta.

---

## 1. Pipeline Cuantizado GGUF (Recomendado para ahorrar VRAM)
Para utilizar el nodo **`[LTX] Distilled GGUF Loader`**, debes colocar los siguientes 6 archivos/directorios en sus respectivas rutas de ComfyUI:

| Nombre del Recurso | Archivo/Directorio Recomendado | Ruta en ComfyUI | Descripción |
| :--- | :--- | :--- | :--- |
| **GGUF Checkpoint** | `ltx-2.3-22b-distilled-Q4_K_M.gguf` | `ComfyUI/models/checkpoints/` | Transformador principal de LTX Video 22B cuantizado en GGUF Q4. |
| **Gemma Directory** | `gemma-3-12b-it-qat-q4_0-unquantized/` | `ComfyUI/models/clip/` | Carpeta de HuggingFace que contiene el codificador de texto de Gemma 3. |
| **Spatial Upscaler** | `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` | `ComfyUI/models/upscale_models/` | Modelo de superresolución espacial x2 para escalar el video a resolución final. |
| **VAE Video** | `ltx-2.3-22b-dev_video_vae.safetensors` | `ComfyUI/models/vae/` | Codificador/Decodificador temporal-espacial de video LTX 2.3. |
| **Audio VAE** | `ltx-2.3-22b-dev_audio_vae.safetensors` | `ComfyUI/models/vae/` | Modelo VAE para decodificación y generación de audio sincronizado (opcional). |
| **Connector** | `ltx-2.3-22b-dev_embeddings_connectors.safetensors` | `ComfyUI/models/clip/` | Proyector de embeddings que conecta el codificador de texto con el transformador. |

> [!NOTE]
> La carpeta de Gemma debe ser una carpeta completa que incluya los archivos de configuración y tokenización (`tokenizer.json`, `tokenizer_config.json`, `config.json`, `model.safetensors.index.json`, etc.), además de los pesos del modelo de texto.

---

## 2. Pipeline Estándar Fast (Nativo SafeTensors)
Para utilizar el nodo **`[LTX] Fast Loader`**, debes colocar los siguientes 3 recursos en tu instalación:

| Nombre del Recurso | Archivo/Directorio Recomendado | Ruta en ComfyUI | Descripción |
| :--- | :--- | :--- | :--- |
| **Checkpoint** | `ltx-2.3-22b-distilled.safetensors` | `ComfyUI/models/checkpoints/` | Transformador de LTX en formato nativo SafeTensors (FP8 o BF16). |
| **Gemma Directory** | `gemma-3-12b-it-qat-q4_0-unquantized/` | `ComfyUI/models/clip/` | Carpeta del codificador de texto Gemma 3. |
| **Spatial Upscaler** | `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` | `ComfyUI/models/upscale_models/` | Modelo de superresolución espacial x2. |

---

## 3. Instrucciones de Carga en ComfyUI

Una vez hayas descargado y colocado los archivos anteriores en sus carpetas indicadas:

1. **Reinicia el servidor de ComfyUI** para limpiar el caché de rutas del sistema `folder_paths`.
2. Los loaders nativos de LTX escanearán estas carpetas y rellenarán los menús desplegables (`COMBO`) de selección con los nombres de archivo exactos que hayas descargado.
3. Si los menús desplegables te aparecen vacíos o con la opción `None`, comprueba que las extensiones coincidan exactamente (`.gguf`, `.safetensors` o directorios válidos en el caso de Gemma).
