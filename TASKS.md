# Plan de Mejoras y Oportunidades

Este documento detalla las tareas para mejorar el código del proyecto `comfyui_flux_collection_advanced`, basándose en las áreas de oportunidad identificadas.

## 1. Refactorización y Limpieza de Código

*   [ ] **Investigar y Resolver el Error de Importación de `PromptConfig`:**
    *   **Descripción:** La clase `PromptConfig` en `application/domain/entities.py` tiene un comentario (`# --- ESTA ES LA CLASE QUE ESTÁ CAUSANDO EL ERROR DE IMPORTACIÓN ---`) que indica un problema.
    *   **Tarea:** Verificar si el error de importación persiste. Si es así, diagnosticar y resolver la causa raíz. Si el problema ya está resuelto, eliminar el comentario para mejorar la claridad del código.
    *   **Archivo:** `application/domain/entities.py`

*   [ ] **Implementar Manejo de Errores Granular con Excepciones de Dominio:**
    *   **Descripción:** Mejorar el manejo de errores introduciendo excepciones personalizadas a nivel de dominio para errores específicos de la lógica de negocio.
    *   **Tarea:**
        *   Definir excepciones personalizadas (ej., `ModelNotFoundError`, `InvalidPromptError`, `ImageSaveError`) en `application/domain/exceptions.py` (crear este archivo si no existe).
        *   Modificar los casos de uso en `application/domain/use_cases.py` para lanzar estas excepciones cuando ocurran condiciones de error específicas del dominio.
        *   Actualizar los adaptadores en `application/infrastructure/comfyui_adapters/` para capturar estas excepciones de dominio y traducirlas a un formato apropiado para ComfyUI (ej., mensajes de error en el log o en la UI).
    *   **Archivos:** `application/domain/use_cases.py`, `application/domain/exceptions.py` (nuevo), `application/infrastructure/comfyui_adapters/*.py`

*   [ ] **Optimizar Logging en `INPUT_TYPES` de Adaptadores:**
    *   **Descripción:** Las llamadas a `logger.info` en los métodos `INPUT_TYPES` de los adaptadores pueden generar un exceso de logs.
    *   **Tarea:** Cambiar `logger.info` a `logger.debug` en los métodos `INPUT_TYPES` de los archivos `flux_models_loader_adapter.py` y `flux_text_prompt_adapter.py`. Evaluar si es necesario mantener el log o si se puede eliminar.
    *   **Archivos:** `application/infrastructure/comfyui_adapters/flux_models_loader_adapter.py`, `application/infrastructure/comfyui_adapters/flux_text_prompt_adapter.py`

## 2. Mejora de la Flexibilidad y Usabilidad

*   [ ] **Refactorizar el Manejo Dinámico de Estilos en `FluxTextPromptRefactored`:**
    *   **Descripción:** Los inputs de estilo (`style1` a `style4`) están codificados de forma rígida, limitando la flexibilidad.
    *   **Tarea:**
        *   Modificar `FluxTextPromptRefactored` para aceptar una lista dinámica de estilos. Esto podría implicar un único campo de texto donde los usuarios ingresen estilos separados por comas, o explorar opciones de UI más avanzadas en ComfyUI si están disponibles para listas dinámicas.
        *   Actualizar la lógica en `execute_refactored` para procesar esta entrada dinámica.
    *   **Archivo:** `application/infrastructure/comfyui_adapters/flux_text_prompt_adapter.py`

*   [ ] **Mejorar la Claridad UI/UX para Subcarpetas en `FluxImageSave`:**
    *   **Descripción:** La lógica de anulación entre `subfolder_name` y `custom_subfolder` puede ser confusa para el usuario.
    *   **Tarea:**
        *   Añadir una descripción más clara en el `tooltip` de `subfolder_name` y `custom_subfolder` en `INPUT_TYPES` para explicar el comportamiento de anulación.
        *   Considerar añadir una nota en la documentación del nodo (`README.MD` o un archivo de documentación específico para el nodo) que detalle cómo funciona la selección de subcarpetas.
    *   **Archivo:** `application/infrastructure/comfyui_adapters/flux_image_save_adapter.py`

## 3. Mantenimiento y Reproducibilidad

*   [ ] **Completar `requirements.txt` con Todas las Dependencias:**
    *   **Descripción:** El archivo `requirements.txt` está vacío, lo que dificulta la reproducibilidad del entorno.
    *   **Tarea:** Identificar todas las dependencias de Python del proyecto (incluyendo `dependency-injector` y cualquier otra librería no estándar de ComfyUI) y listarlas en `requirements.txt` con sus versiones específicas si es posible.
    *   **Archivo:** `requirements.txt`

## 4. Consideraciones Futuras (No Urgentes)

*   [ ] **Evaluar el Tipo de Retorno de `get_available_output_subfolders`:**
    *   **Descripción:** El método `get_available_output_subfolders` en `IImageRepository` devuelve `List[str]`. Si en el futuro se necesitan más atributos para las subcarpetas, un tipo de retorno más estructurado sería beneficioso.
    *   **Tarea:** Mantener esta consideración en mente para futuras extensiones. No se requiere acción inmediata, pero es una oportunidad para una mayor robustez si los requisitos cambian.
    *   **Archivo:** `application/domain/ports.py`
