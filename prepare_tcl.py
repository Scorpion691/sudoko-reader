"""PyInstaller helper: some Python 3.13+ installs keep Tcl/Tk 9 libraries inside zip files (D:\\tcl\\lib*.zip).
PyInstaller then bundles no _tcl_data / _tk_data and the .exe dies at start. This unpacks them to build_tcl/."""
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "build_tcl")
found = []
for base in {os.path.join(sys.base_prefix, "tcl"), os.path.join(os.path.dirname(sys.executable), "tcl")}:
    if not os.path.isdir(base):
        continue
    for f in os.listdir(base):
        if f.startswith(("libtcl", "libtk")) and f.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(base, f)) as z:
                z.extractall(OUT)
            found.append(f)
print("unpacked:", found or "nothing (not needed on this Python)")
