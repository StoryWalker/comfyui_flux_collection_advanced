# Informe Técnico: Flux Collection Advanced (v1.0.0)
Ubicación: `team_pro/docs/update_context_v1.0.0.md`

## 1. Objetivo del Proyecto
Proporcionar una colección de nodos avanzados para ComfyUI que automaticen la complejidad de la arquitectura FLUX, permitiendo el uso fluido de modelos GGUF, el manejo inteligente de condicionamiento y la decodificación robusta de imágenes y video.

## 2. Highlights (Logros Significativos)
- **Autodetección de Arquitectura:** El nodo de muestreo detecta automáticamente si el modelo requiere 16 o 128 canales latentes y ajusta la generación en consecuencia.
- **Mecanismo de Padding de Condicionamiento:** Corrige errores de dimensión en los prompts (ej. de 4096 a 6144) mediante padding dinámico con ceros.
- **Soporte GGUF y Parcheo de Muestreo:** Aplicación automática del shift de 1.15 necesario para que los modelos Flux generen resultados óptimos.
- **Decodificación Versátil:** Manejo de tensores 5D para video y corrección de canales en el VAE.

## 3. Puntos Clave y Arquitectura
- **Estructura de Carga:** Separación de cargadores de modelos estándar y GGUF, con soporte explícito para variantes FP8.
- **Gestión de TAESD:** Lógica integrada para cargar y escalar variantes rápidas de VAE (TAESD).
- **ControlNet Optimizado:** Flujo de aplicación de ControlNet adaptado al condicionamiento de Flux.

## 4. Stack Detectado
- **Python:** 3.10+ (entorno ComfyUI)
- **Torch:** ~2.x (con soporte FP8)
- **Librerías:** `numpy`, `Pillow`, `comfyui-gguf` (plugin externo requerido).

---
*Generado por Gemini CLI para TEAM_PRO - 19 de febrero de 2026*
