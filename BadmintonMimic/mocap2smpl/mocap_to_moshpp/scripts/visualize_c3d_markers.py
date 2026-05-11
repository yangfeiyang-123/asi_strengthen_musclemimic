from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts" / "visualize_c3d_markers.py"), run_name="__main__")
