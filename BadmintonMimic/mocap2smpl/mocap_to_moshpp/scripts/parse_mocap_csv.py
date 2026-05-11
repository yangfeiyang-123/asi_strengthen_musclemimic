from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts" / "parse_mocap_csv.py"), run_name="__main__")
