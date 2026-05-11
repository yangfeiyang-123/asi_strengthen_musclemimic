from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[2] / "scripts" / "prepare_moshpp_config.py"), run_name="__main__")
