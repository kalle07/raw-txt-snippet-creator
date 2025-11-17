import sys
import subprocess
import os
from pathlib import Path

# Define the entry point (your main script)
entry_point = "snippet_extractor_v08alpha.py"

# Path to your image
image_file = "example.bmp"

# Build command with PyInstaller arguments
cmd = [
    sys.executable,
    "-m", "PyInstaller",
    "--onefile",
    "--noconfirm",
    "--clean",
    "--noconsole",
    "--hidden-import", "wx",
    "--hidden-import", "rapidfuzz",
    "--hidden-import", "typing_extensions",
    # Include the image
    "--add-data", f"{image_file}{os.pathsep}.",
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
