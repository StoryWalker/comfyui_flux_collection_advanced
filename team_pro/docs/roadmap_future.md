# Roadmap de Desarrollo Futuro: Flux Collection Advanced

Este documento detalla las áreas de oportunidad estratégicas para la evolución de la colección de nodos, enfocándose en la eficiencia de hardware y la expansión de capacidades generativas.

## 1. Gestión de VRAM para T5 (Optimización Crítica - EN PROGRESO)
- **Objetivo:** Minimizar el impacto del encoder T5-XXL en GPUs con menos de 16GB de VRAM.
- **Acción:** Implementar nodos con sufijo `_beta` para testear el truncado de capas y el offloading agresivo. (Nodo `FluxVRAMLoaderBeta` funcional).

## 2. Soporte LoRA GGUF-Optimized
- **Objetivo:** Permitir el uso de múltiples LoRAs sobre modelos cuantizados sin pérdida significativa de velocidad.
- **Acción:** Desarrollar un "LoRA Stacker" que se integre con la lógica de dequantización dinámica.

## 3. Planificador de Guidance (Guidance Scheduling)
- **Objetivo:** Mejorar la coherencia y el detalle fino permitiendo curvas dinámicas de guidance durante el muestreo.
- **Acción:** Crear nodos de control que modifiquen el `guidance_scale` paso a paso.

## 4. Extractor de Metadatos GGUF (COMPLETADO)
- **Objetivo:** Facilitar la configuración del usuario mediante la lectura automática de la resolución de entrenamiento y el autor del modelo.
- **Acción:** Nodo informativo `FluxGGUFInfo` que expone los metadatos internos del formato GGUF. (Finalizado 2026-02-19).

## 5. Especialización en Arquitecturas de Video
- **Objetivo:** Liderar el soporte para modelos DiT de video como Wan 2.1 y Hunyuan Video.
- **Acción:** Expandir el soporte de latentes 5D y condicionamiento temporal.

---
*Documento de seguimiento para TEAM_PRO - 19 de febrero de 2026*
