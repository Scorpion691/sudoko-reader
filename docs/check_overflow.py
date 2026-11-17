"""Reports which report pages have content that does not fit on the page:  python docs/check_overflow.py"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_docs import browser  # noqa: E402

name = sys.argv[1] if len(sys.argv) > 1 else "Report.html"
b = browser()
uri = "file:///" + os.path.join(HERE, name).replace("\\", "/") + "?check"
out = subprocess.run([b, "--headless", "--disable-gpu", "--virtual-time-budget=10000", "--dump-dom", uri],
                     capture_output=True, text=True, encoding="utf-8", timeout=120).stdout
m = re.search(r"OVERFLOW=(\[.*?\])", out)
print("overflow (page:px past the footer):", m.group(1) if m else "check did not run")
