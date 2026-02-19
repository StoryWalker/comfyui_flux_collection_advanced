# Guía de Configuración: Flux VRAM Loader (BETA) - REVISIÓN 2
Ubicación: `team_pro/docs/reference_loader_config.md`

Este documento refleja el orden exacto de los parámetros tras la refactorización de estabilidad para evitar errores de validación.

---

## Orden de Parámetros en la Interfaz (Actualizado)

1. **unet_name**: Selección del modelo principal (.gguf recomendado para ahorro).
2. **dequant_dtype**: Tipo de dato para dequantización (usar `float16` o `bfloat16`).
3. **clip_name1**: Primer encoder de texto (CLIP-L).
4. **clip_name2**: Segundo encoder de texto (T5-XXL). Usar **`None`** para modo Flux 2.
5. **t5_optimization**: Estrategia de ahorro (`Layer Truncation` recomendada).
6. **vae_name**: Selección del VAE o variante TAESD.
7. **t5_layers**: Cantidad de capas de T5 a procesar. **16** es el balance ideal.

---

## Configuraciones Maestras

### A. Flux 2 + GGUF (Máximo Ahorro)
*Ideal para GPUs con 8GB-12GB VRAM.*
- **unet_name**: `modelo_flux2.gguf`
- **dequant_dtype**: `float16`
- **clip_name2**: **`None`**
- **vae_name**: `taef1` (o tu VAE favorito)
- **t5_layers**: `16`

### B. Flux 1 + GGUF (Calidad con Ahorro)
*Ideal para mantener la precisión del T5 en hardware ajustado.*
- **unet_name**: `modelo_flux1.gguf`
- **clip_name2**: `t5xxl_fp8.safetensors`
- **t5_optimization**: `Layer Truncation`
- **t5_layers**: **`12`** o **`16`**

---

## Notas Técnicas de Estabilidad
- Si el nodo muestra errores de "INT value", asegúrate de que no haya cables cruzados de versiones anteriores. Borrar y volver a crear el nodo en el canvas soluciona cualquier desajuste de widgets.
- El parámetro `t5_layers` ahora es robusto ante entradas de texto accidentales.

---
*Documentación actualizada - TEAM_PRO - 19 de febrero de 2026*
