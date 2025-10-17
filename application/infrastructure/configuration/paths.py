# /application/infrastructure/configuration/paths.py
# 1.0.0 - Initial version. Defines core project paths.

from pathlib import Path
import os

class Paths:
    """
    A centralized class for managing all important filesystem paths for the application.

    This class ensures that all parts of the application refer to paths in a
    consistent and reliable way, making the system more maintainable and easier
    to reconfigure. It follows the Single Responsibility Principle (SRP) by
    solely managing path definitions.
    """

    def __init__(self, base_dir: Path = None):
        """
        Initializes the Paths object.

        Args:
            base_dir (Path, optional): The base directory of the project. 
                                       If None, it defaults to the parent directory
                                       of this file's location.
        """
        # The root of the project is considered the 'application' folder's parent
        self.project_root: Path = base_dir if base_dir else Path(__file__).resolve().parents[3]
        
        # Main 'application' directory
        self.application_root: Path = self.project_root / "application"

        # --- Subdirectories ---
        self.data: Path = self.application_root / "data"
        self.input_data: Path = self.data / "input"
        self.output_data: Path = self.data / "output"
        
        self.docs: Path = self.application_root / "docs"
        self.domain: Path = self.application_root / "domain"
        self.infrastructure: Path = self.application_root / "infrastructure"
        self.scripts: Path = self.application_root / "scripts"
        self.tests: Path = self.application_root / "tests"

        # --- Infrastructure specific paths ---
        self.configuration: Path = self.infrastructure / "configuration"
        self.repositories: Path = self.infrastructure / "repositories"
        self.services: Path = self.infrastructure / "services"
        
        # --- Data files ---
        # We can also define specific file paths here
        self.styles_csv: Path = self.input_data / "styles.csv"

    def ensure_all_dirs_exist(self) -> None:
        """
        Ensures that all defined directory paths exist, creating them if necessary.

        This is useful for application startup to prevent FileNotFoundError exceptions.
        """
        directories = [
            self.data, self.input_data, self.output_data, self.docs, self.domain,
            self.infrastructure, self.scripts, self.tests, self.configuration,
            self.repositories, self.services
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        print("All necessary directories are ensured to exist.")

# --- Global instance for easy access ---
# This instance can be imported by other parts of the application.
# Dependency injection is a better pattern for providing this configuration,
# but a global instance is a practical starting point.
paths = Paths()