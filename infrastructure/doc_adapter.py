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
        # Limpiar y estructurar el docstring
        lines = [line.strip() for line in docstring.split('\n') if line.strip()]
        clean_doc = "<br>".join(lines)
        cls.DESCRIPTION = f"<div align='center'><b>[HEX] Arquitectura Avanzada</b></div><br>{clean_doc}<br><br><i>{HEX_AI_FOOTER}</i>"
    else:
        cls.DESCRIPTION = f"<div align='center'><b>[HEX] Arquitectura Avanzada</b></div><br>Nodo de la arquitectura Hexagonal.<br><br><i>{HEX_AI_FOOTER}</i>"
        
    return cls
