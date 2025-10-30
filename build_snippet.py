import sys
import subprocess
import os
from pathlib import Path

# Define the entry point (your main script)
entry_point = "snippet_extracor_v03alpha.py"

# Build command with PyInstaller arguments
cmd = [
    sys.executable,
    "-m", "PyInstaller",
    "--onefile",
    "--noconfirm",
    "--clean",
#    "--console",  # No console window (important for GUI applications)
    "--hidden-import", "wx",
    "--hidden-import", "numpy",
    "--hidden-import", "rapidfuzz",
    "--hidden-import", "typing_extensions",
]

# Add the entry point
cmd.append(entry_point)

# Execute the build command
try:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print("Compilation completed successfully.")
    
    # Show any warnings or info from PyInstaller
    if result.stderr:
        for line in result.stderr.split('\n'):
            if line.strip() and not line.startswith('['):
                print(f"PyInstaller: {line}")
                
except subprocess.CalledProcessError as e:
    print(f"Error during compilation: {e}")
    print("Stderr:", e.stderr)
    
except FileNotFoundError:
    print("PyInstaller not found. Please install it with 'pip install pyinstaller'")
