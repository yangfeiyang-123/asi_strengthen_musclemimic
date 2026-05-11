from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts" / "convert_moshpp_output.py"), run_name="__main__")
