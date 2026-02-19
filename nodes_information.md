# Reporte Técnico: Arquitectura de Nodos y Ecosistema FLUX en ComfyUI

Este documento sirve como base de conocimiento para el desarrollo de la colección **comfyui_flux_collection_advanced**. Contiene el análisis de dependencias, estándares de la API y lógica específica de FLUX extraída del núcleo de ComfyUI (`nodes.py`) y sus extensiones oficiales.

---

## 1. Fundamentos de la Arquitectura de ComfyUI

ComfyUI es un motor de ejecución basado en grafos donde cada nodo es una unidad lógica independiente.

### Mecanismos de Registro de Nodos
Existen dos formas de exponer nodos al sistema:
1.  **V1 (Legacy/Estándar):** Definición manual de `INPUT_TYPES`, `RETURN_TYPES`, y `FUNCTION`. Se registran en un diccionario global `NODE_CLASS_MAPPINGS`.
2.  **V3 (API Moderna):** Uso de `comfy_api.latest.io` y herencia de `io.ComfyNode`. Es más robusto para validación de tipos y permite una integración más limpia con la UI moderna. Utiliza un `comfy_entrypoint` asíncrono.

### El Ciclo de Vida del Nodo
- **Descubrimiento:** ComfyUI escanea `custom_nodes/` buscando archivos `.py` o carpetas con `__init__.py`.
- **Validación:** Verifica la presencia de `NODE_CLASS_MAPPINGS` o `comfy_entrypoint`.
- **Ejecución:** El motor (`execution.py`) ordena los nodos y llama al método definido en `FUNCTION` (V1) o `execute` (V3), inyectando las dependencias necesarias.

---

## 2. Árbol de Dependencias Críticas

Para el desarrollo de nodos avanzados, se deben importar y utilizar los siguientes módulos internos:

| Módulo | Propósito Técnico |
| :--- | :--- |
| `comfy.model_management` | **Crucial.** Gestiona la VRAM, carga/descarga de modelos (`get_torch_device`), y control de interrupciones (`throw_exception_if_processing_interrupted`). |
| `folder_paths` | Gestiona las rutas del sistema. Permite acceder a `checkpoints`, `loras`, `vae`, etc., de forma agnóstica a la instalación del usuario. |
| `node_helpers` | Proporciona utilidades para manipular condicionamientos y configuraciones de nodos de forma estandarizada. |
| `comfy.utils` | Funciones para reescalado de imágenes (`common_upscale`), gestión de tensores y carga de archivos. |
| `comfy.sd` | Contiene las clases base para los modelos (Stable Diffusion, FLUX, etc.) y sus respectivos cargadores. |

---

## 3. Especificaciones Técnicas para FLUX

FLUX presenta diferencias arquitectónicas críticas comparado con SD1.5/SDXL que deben ser respetadas en los nuevos nodos:

### A. Estructura de Tensores Latentes
- **Factor de Compresión:** 16x (SD es 8x). Un input de `1024x1024` genera un latente de `64x64`.
- **Canales Latentes:** Utiliza **128 canales** (en lugar de los 4 habituales de SD).
- **Formato:** `[Batch, 128, H//16, W//16]`.

### B. Condicionamiento (Conditioning)
- **Codificación Dual:** Requiere `clip_l` (CLIP ViT-L/14) y `t5xxl` (T5-v1.1-XXL). 
- **Guidance:** FLUX utiliza un parámetro de guía inyectado en el condicionamiento. El valor estándar es `3.5`. Se aplica mediante:
  `node_helpers.conditioning_set_values(conditioning, {"guidance": valor})`.
- **T5XXL:** El nodo de codificación debe ser capaz de procesar el texto específicamente para T5 para evitar truncamientos y asegurar la calidad del prompt.

### C. Schedulers y Sigmas
- **SNR Shift:** FLUX funciona mejor con un "shift" en los timesteps basado en la resolución. A mayor resolución, se requiere un ajuste en la curva de ruido para evitar artefactos.
- **Cálculo de Seq_Len:** `(Width * Height) / (16 * 16)`. Este valor se usa para calcular el `mu` empírico del scheduler.

---

## 4. Guía de Implementación para "comfyui_flux_collection_advanced"

### Estándares de Codificación
1.  **Formato de Imagen:** Siempre manejar tensores en formato **NHWC** `[B, H, W, C]` con valores flotantes `[0.0, 1.0]`.
2.  **Dispositivo (Device):** Nunca asumir `cuda:0`. Usar `comfy.model_management.get_torch_device()` o `comfy.model_management.intermediate_device()`.
3.  **Memoria:** Si el nodo realiza operaciones pesadas, llamar a `comfy.model_management.soft_empty_cache()` al finalizar si es necesario, aunque el motor suele gestionarlo.
4.  **Validación:** Implementar `check_inputs` o validaciones dentro de `execute` para dar errores claros (ej: "Input image must be a multiple of 16 for FLUX").

### Ejemplo de Estructura de Nodo Avanzado (V3)
```python
from comfy_api.latest import io
import comfy.model_management
import node_helpers

class FluxAdvancedNode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="FluxAdvancedNode",
            category="advanced/flux",
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Float.Input("guidance_override", default=3.5)
            ],
            outputs=[io.Conditioning.Output()]
        )

    @classmethod
    def execute(cls, conditioning, guidance_override):
        # Lógica de manipulación usando node_helpers
        new_cond = node_helpers.conditioning_set_values(
            conditioning, {"guidance": guidance_override}
        )
        return io.NodeOutput(new_cond)
```

---

## 5. Notas Relevantes para Futuras IAs
- **Entorno:** ComfyUI corre típicamente en Python 3.10+ con PyTorch.
- **Plugins:** Los nodos personalizados pueden incluir su propia carpeta `web/` con archivos `.js` para modificar el comportamiento del frontend (widgets dinámicos, previsualizaciones).
- **Asincronía:** Los nodos V3 soportan ejecución asíncrona, útil para llamadas a APIs o procesos de E/S lentos.
