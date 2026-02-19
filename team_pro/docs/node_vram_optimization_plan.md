# Plan de Optimización: Manejo de VRAM (Flux 2 + GGUF) - ACTUALIZADO
Ubicación: `team_pro/docs/node_vram_optimization_plan.md`

## Contexto Técnico Actualizado
Se ha pivotado el nodo Beta para especializarse en la arquitectura **Flux 2** y el formato **GGUF**, buscando la máxima eficiencia posible en ComfyUI.

## Implementación: `FluxModelsLoader_VRAM_Beta`

### 1. Soporte GGUF Dinámico
- Integración con `UnetLoaderGGUFAdvanced`.
- Soporte para `dequant_dtype` y `patch_dtype`.

### 2. Modo Flux 2 (Single CLIP)
- Bypass inteligente del segundo encoder (T5).
- Reducción masiva de VRAM al eliminar el encoder más pesado si el modelo lo permite.

### 3. Truncamiento de T5 (Dual Mode)
- Para configuraciones que requieren T5, se permite truncar las capas del encoder (default 16/24).

### 4. Sampling Shift 1.15
- Inyectado automáticamente para corregir la curva de ruido en Flux 1/2.

---
*Documentación actualizada - TEAM_PRO*
