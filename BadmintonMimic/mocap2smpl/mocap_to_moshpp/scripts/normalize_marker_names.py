from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts" / "normalize_marker_names.py"), run_name="__main__")
