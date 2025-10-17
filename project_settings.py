# /project_settings.py
# 0.1.0 - Initial version for project scaffolding.

import os

def create_project_structure():
    """
    Creates the standard project directory structure based on Hexagonal Architecture.
    """
    base_path = "application"
    
    # Core directories as per the specified structure
    dirs_to_create = [
        "data/input",
        "data/output",
        "docs",
        "domain",
        "include",
        "infrastructure/configuration",
        "infrastructure/dependency_injection",
        "infrastructure/repositories",
        "infrastructure/services",
        "infrastructure/web",
        "scripts",
        "tests/domain",
        "tests/infrastructure/repositories",
        "tests/integration",
    ]

    # Create directories
    for d in dirs_to_create:
        os.makedirs(os.path.join(base_path, d), exist_ok=True)

    # Core files to create with __init__.py markers
    files_to_create = {
        "domain/__init__.py": "",
        "domain/entities.py": "# Defines core domain entities",
        "domain/ports.py": "# Defines interfaces (ports) for infrastructure interaction",
        "domain/use_cases.py": "# Implements application use cases or business logic",
        "infrastructure/__init__.py": "",
        "infrastructure/configuration/__init__.py": "",
        "infrastructure/configuration/paths.py": "# Defines important filesystem paths",
        "infrastructure/configuration/settings.py": "# Stores application configurations",
        "infrastructure/dependency_injection/__init__.py": "",
        "infrastructure/dependency_injection/container.py": "# Defines and configures the DI container",
        "infrastructure/repositories/__init__.py": "",
        "infrastructure/services/__init__.py": "",
        "infrastructure/web/__init__.py": "",
        "tests/domain/__init__.py": "",
        "tests/infrastructure/__init__.py": "",
        "tests/integration/__init__.py": "",
    }

    # Create files
    for file_path, content in files_to_create.items():
        full_path = os.path.join(base_path, file_path)
        if not os.path.exists(full_path):
            with open(full_path, "w") as f:
                f.write(f"# {os.path.basename(full_path)}\n{content}\n")

    print("Project structure created successfully inside 'application/' folder.")

if __name__ == "__main__":
    create_project_structure()