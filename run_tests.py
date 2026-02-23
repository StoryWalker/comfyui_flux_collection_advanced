# -*- coding: utf-8 -*-
import unittest
import sys
import os

# Configurar el entorno para que reconozca los módulos del proyecto
# Añadimos el directorio padre de 'custom_nodes' si fuera posible, 
# pero como estamos restringidos, añadimos la raíz actual.
sys.path.append(os.getcwd())

def run():
    # Descubrir y ejecutar tests
    loader = unittest.TestLoader()
    suite = loader.discover('tests', pattern='test_*.py')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Salir con código de error si las pruebas fallan
    if not result.wasSuccessful():
        sys.exit(1)

if __name__ == "__main__":
    run()
