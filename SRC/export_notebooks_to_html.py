"""
export_notebooks_to_html.py

Converts all 23 project notebooks into rich, interactive HTML reports
for browser walkthroughs and GitHub artifact releases.
"""

import subprocess
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = WORKSPACE / "NOTEBOOKS"
HTML_OUTPUT_DIR = WORKSPACE / "RESULTS" / "executed_notebooks_html"
HTML_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
print(f"Discovered {len(notebooks)} notebooks to export to HTML.")

success_count = 0
for nb_path in notebooks:
    out_file = HTML_OUTPUT_DIR / f"{nb_path.stem}.html"
    print(f"Exporting {nb_path.name} -> {out_file.name}...")
    cmd = [
        sys.executable, "-m", "nbconvert",
        "--to", "html",
        "--output-dir", str(HTML_OUTPUT_DIR),
        str(nb_path)
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0:
        success_count += 1
        print(f"  Successfully exported: {out_file.name} ({out_file.stat().st_size // 1024} KB)")
    else:
        print(f"  Failed: {res.stderr.strip()}")

print(f"\nHTML Export Summary: {success_count}/{len(notebooks)} exported successfully to {HTML_OUTPUT_DIR}")
