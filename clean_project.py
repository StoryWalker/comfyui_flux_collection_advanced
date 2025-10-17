# /clean_project.py
__version__ = "1.1.0"
# 1.1.0 - Allow passing start_path as a command-line argument.
# 1.0.0 - A utility script to find and delete all __pycache__ directories.

import os
import shutil
import sys # Required to read command-line arguments

def clean_pycache(start_path: str = '.') -> None:
    """
    Recursively finds and removes all __pycache__ directories within a given path.

    Args:
        start_path (str): The starting directory to search from. Defaults to the
                          current directory.
    """
    deleted_count = 0
    print(f"Starting cleanup from root: {os.path.abspath(start_path)}\n")

    # os.walk efficiently traverses the directory tree top-down
    for root, dirs, files in os.walk(start_path):
        # We check for '__pycache__' in the list of directories found
        if '__pycache__' in dirs:
            pycache_path = os.path.join(root, '__pycache__')
            print(f"Found: {pycache_path}")
            
            try:
                # shutil.rmtree is a powerful tool to remove a directory and all its contents
                shutil.rmtree(pycache_path)         
                print(f"--> DELETED\n")
                deleted_count += 1
            except OSError as e:
                print(f"--> ERROR deleting {pycache_path}: {e}\n")

    if deleted_count > 0:
        print(f"✨ Cleanup complete. Successfully deleted {deleted_count} `__pycache__` director(y/ies).")
    else:
        print("✅ No `__pycache__` directories found. Project is clean.")


if __name__ == "__main__":
    # Check if a path was provided as an argument
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
        if not os.path.isdir(root_path):
            print(f"Error: Provided path '{root_path}' is not a valid directory.")
            sys.exit(1)
        clean_pycache(root_path)
    else:
        # If no argument, run in the current directory
        clean_pycache()