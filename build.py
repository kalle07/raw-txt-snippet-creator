"""Build the Text Search desktop application with PyInstaller.

Run this script from the project directory with:

    python build.py

The finished Windows application is written to ``dist/TextSearch-by-kalle07.exe``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
ENTRY_POINT = PROJECT_ROOT / "start.py"
APP_NAME = "TextSearch-by-kalle07"


def main() -> None:
    """Package the wxPython text-search application as a single Windows executable."""
    if not ENTRY_POINT.is_file():
        raise FileNotFoundError(f"Application entry point not found: {ENTRY_POINT}")

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onefile",
        "--windowed",
        "--name",
        APP_NAME,
        "--distpath",
        str(PROJECT_ROOT / "dist"),
        "--workpath",
        str(PROJECT_ROOT / "build"),
        "--specpath",
        str(PROJECT_ROOT / "build"),
        "--paths",
        str(PROJECT_ROOT),
        # wxPython submodules used by the GUI.
        "--hidden-import",
        "wx.lib.newevent",
        # engine.py is intentionally imported after the GUI is displayed,
        # but its dependencies still need to be bundled by PyInstaller.
        "--hidden-import",
        "lancedb",
        "--hidden-import",
        "lancedb.query",
        "--hidden-import",
        "numpy",
        "--hidden-import",
        "pyarrow",
        "--hidden-import",
        "rapidfuzz",
        str(ENTRY_POINT),
    ]

    print(f"Building {APP_NAME}.exe...")
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    output = PROJECT_ROOT / "dist" / f"{APP_NAME}.exe"
    print(f"Build complete: {output}")


if __name__ == "__main__":
    main()
