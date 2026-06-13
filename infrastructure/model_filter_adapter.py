import os
import json
import folder_paths

# Singleton pattern para cachear los filtros en memoria
_FILTERS_CACHE = None
_FILTERS_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "model_filters.json")

def get_filters():
    global _FILTERS_CACHE
    if _FILTERS_CACHE is None:
        if os.path.exists(_FILTERS_PATH):
            try:
                with open(_FILTERS_PATH, "r", encoding="utf-8") as f:
                    _FILTERS_CACHE = json.load(f)
            except Exception as e:
                print(f"[ModelFilterAdapter] Error cargando {_FILTERS_PATH}: {e}")
                _FILTERS_CACHE = {}
        else:
            _FILTERS_CACHE = {}
    return _FILTERS_CACHE

def get_filtered_filenames(node_class_name: str, field_name: str, folder_type: str) -> list:
    """
    Obtiene la lista de archivos de folder_paths y aplica los filtros JSON.
    Si no hay filtro para este nodo/campo, retorna la lista original.
    """
    original_list = folder_paths.get_filename_list(folder_type)
    filters = get_filters()
    
    # Navegar por el JSON: node_class_name -> field_name
    node_filters = filters.get(node_class_name, {})
    field_rules = node_filters.get(field_name, {})
    
    if not field_rules:
        return original_list
        
    include_extensions = field_rules.get("include_extensions", [])
    exclude_extensions = field_rules.get("exclude_extensions", [])
    include_keywords = field_rules.get("include_keywords", [])
    exclude_keywords = field_rules.get("exclude_keywords", [])
    
    filtered_list = []
    for filename in original_list:
        filename_lower = filename.lower()
        
        # 1. Filtro de extensiones
        if include_extensions:
            if not any(filename_lower.endswith(ext.lower()) for ext in include_extensions):
                continue
                
        if exclude_extensions:
            if any(filename_lower.endswith(ext.lower()) for ext in exclude_extensions):
                continue
                
        # 2. Filtro de palabras clave (inclusivo: debe tener al menos una si se especifica)
        if include_keywords:
            if not any(kw.lower() in filename_lower for kw in include_keywords):
                continue
                
        # 3. Filtro de palabras clave (exclusivo: se descarta si tiene alguna)
        if exclude_keywords:
            if any(kw.lower() in filename_lower for kw in exclude_keywords):
                continue
                
        filtered_list.append(filename)
        
    # Si por alguna razón el filtro fue demasiado estricto y dejó la lista vacía,
    # regresamos la original para evitar crashes en ComfyUI
    if not filtered_list and original_list:
        print(f"[ModelFilterAdapter] Warning: Filtro vació la lista para {node_class_name}.{field_name}. Retornando lista original.")
        return original_list
        
    return filtered_list
