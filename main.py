#!/usr/bin/env python3
"""
Study Assistant - Self-bootstrapping entry point

This script handles environment setup and launches the main application.
It automatically:
- Validates Python version (3.8+ required)
- Creates a virtual environment if needed
- Installs dependencies
- Relaunches itself in the virtual environment

Usage:
    python3 main.py [--reset]

Options:
    --reset     Delete virtual environment and reinstall dependencies
"""

import sys
import subprocess
import os
from pathlib import Path


# ============================================================================
# Constants
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.resolve()
VENV_DIR = PROJECT_ROOT / ".venv"
PYTHON_MIN_VERSION = (3, 8)
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"


# ============================================================================
# Bootstrap Functions
# ============================================================================

def check_python_version() -> None:
    """
    Validate that we're running on Python 3.8 or higher.
    
    Exits with error code 1 if version is insufficient.
    """
    if sys.version_info < PYTHON_MIN_VERSION:
        major, minor = PYTHON_MIN_VERSION
        current_major, current_minor = sys.version_info.major, sys.version_info.minor
        
        print(f"Error: Python {major}.{minor}+ is required")
        print(f"You are running: Python {current_major}.{current_minor}")
        print(f"\nPlease install Python {major}.{minor} or higher from:")
        print("https://www.python.org/downloads/")
        sys.exit(1)


def get_venv_python() -> Path:
    """
    Get the path to the Python executable inside the virtual environment.
    
    Returns:
        Path to the venv Python executable
    """
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def is_in_venv() -> bool:
    """
    Check if the current Python process is running inside a virtual environment.
    
    Returns:
        True if in venv, False otherwise
    """
    # Check for virtualenv
    if hasattr(sys, 'real_prefix'):
        return True
    
    # Check for venv
    if hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
        return True
    
    return False


def create_venv() -> None:
    """
    Create a virtual environment in the .venv directory.
    
    Exits with error code 1 if creation fails.
    """
    if VENV_DIR.exists():
        return
    
    print("Creating virtual environment...")
    print(f"Location: {VENV_DIR}")
    
    try:
        subprocess.check_call(
            [sys.executable, "-m", "venv", str(VENV_DIR)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        print("Virtual environment created successfully\n")
    except subprocess.CalledProcessError as e:
        print(f"\nError: Failed to create virtual environment")
        print(f"Details: {e.stderr.decode() if e.stderr else 'Unknown error'}")
        print("\nTry installing venv:")
        print("  Ubuntu/Debian: sudo apt-get install python3-venv")
        print("  macOS: Should be included with Python")
        print("  Windows: Should be included with Python")
        sys.exit(1)


def check_dependencies_installed() -> bool:
    """
    Check if required dependencies are already installed in the venv.
    
    Returns:
        True if key packages are installed, False otherwise
    """
    venv_python = get_venv_python()
    
    if not venv_python.exists():
        return False
    
    try:
        # Run pip list and check for key packages
        result = subprocess.run(
            [str(venv_python), "-m", "pip", "list", "--format=freeze"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10
        )
        
        installed_packages = result.stdout.lower()
        
        # Check for critical packages
        required_packages = ["chromadb", "langchain", "rich"]
        return all(pkg in installed_packages for pkg in required_packages)
        
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


def install_dependencies() -> None:
    """
    Install dependencies from requirements.txt into the virtual environment.
    
    Exits with error code 1 if installation fails.
    """
    if not REQUIREMENTS_FILE.exists():
        print(f"Error: {REQUIREMENTS_FILE} not found")
        sys.exit(1)
    
    # Check if already installed
    if check_dependencies_installed():
        return
    
    print("Installing dependencies...")
    print("This may take a few minutes on first run.\n")
    
    venv_python = get_venv_python()
    
    try:
        # Upgrade pip first
        subprocess.check_call(
            [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Install requirements
        subprocess.check_call(
            [str(venv_python), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )
        
        print("Dependencies installed successfully\n")
        
    except subprocess.CalledProcessError as e:
        print(f"\nError: Failed to install dependencies")
        print(f"Details: {e.stderr.decode() if e.stderr else 'Unknown error'}")
        print("\nTry installing manually:")
        print(f"  {venv_python} -m pip install -r {REQUIREMENTS_FILE}")
        sys.exit(1)


def relaunch_in_venv() -> None:
    """
    Relaunch this script using the virtual environment's Python interpreter.
    
    This function does not return - it replaces the current process.
    """
    venv_python = get_venv_python()
    
    if not venv_python.exists():
        print(f"Error: Virtual environment Python not found at: {venv_python}")
        sys.exit(1)
    
    print("Relaunching in virtual environment...\n")
    
    # Replace current process with venv Python running this script
    # Pass through any command-line arguments
    os.execv(str(venv_python), [str(venv_python), __file__] + sys.argv[1:])


def handle_reset() -> None:
    """
    Handle the --reset flag by deleting the virtual environment.
    """
    import shutil
    
    if VENV_DIR.exists():
        print("Removing virtual environment...")
        shutil.rmtree(VENV_DIR)
        print("Reset complete. Run again to reinstall.\n")
    else:
        print("No virtual environment found to reset.\n")
    
    sys.exit(0)


def bootstrap() -> None:
    """
    Main bootstrap logic.
    
    Ensures the application runs in a properly configured virtual environment
    with all dependencies installed.
    """
    # Handle --reset flag
    if "--reset" in sys.argv:
        handle_reset()
    
    # Check Python version first
    check_python_version()
    
    # If already in venv, we're done bootstrapping
    if is_in_venv():
        return
    
    # Not in venv - set it up
    create_venv()
    install_dependencies()
    relaunch_in_venv()


# ============================================================================
# Main Application
# ============================================================================

def main() -> None:
    """
    Main application entry point.
    
    This only runs after successful bootstrap (i.e., in the venv).
    """
    try:
        # Import main CLI here (after bootstrap ensures dependencies are available)
        from src.cli import run_application
        
        # Launch the application
        run_application()
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    # Bootstrap the environment
    bootstrap()
    
    # Run the main application
    main()
