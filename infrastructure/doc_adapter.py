import inspect

HEX_AI_FOOTER = "This documentation was AI-generated. If you find any errors or have suggestions for improvement, please feel free to contribute! Edit on GitHub."

def hex_node_doc(cls):
    """
    Decorador para nodos HEX de ComfyUI.
    Extrae el docstring de la clase, lo limpia y le añade el pie de página de IA.
    Inyecta el resultado en el atributo cls.DESCRIPTION para que ComfyUI lo lea.
    """
    docstring = inspect.getdoc(cls)
    
    if docstring:
        # Limpiar y estructurar el docstring
        lines = [line.strip() for line in docstring.split('\n') if line.strip()]
        clean_doc = "\n\n".join(lines)
        cls.DESCRIPTION = f"[HEX] Arquitectura Avanzada\n\n{clean_doc}\n\n{HEX_AI_FOOTER}"
    else:
        cls.DESCRIPTION = f"[HEX] Arquitectura Avanzada\n\nNodo de la arquitectura Hexagonal.\n\n{HEX_AI_FOOTER}"
        
    return cls
