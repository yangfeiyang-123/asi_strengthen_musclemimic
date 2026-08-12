import importlib.util
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


SAMPLE_CSV = """# participant_id,yfy
# action_label,high_clear
# movement_quality,correct
# error_label,correct
# trial_index,1
# timestamp,20260622_204138
# sample_rate_hz,2000
# sensor_channels,1 2
time_s,sensor_1_Gastrocnemius_mV,sensor_2_Gluteus_Maximus_mV
0.000000,0.10000000,0.20000000
0.000500,0.30000000,0.40000000
"""


class VisualizeScriptTest(unittest.TestCase):
    def load_script(self):
        script_path = Path(__file__).resolve().parents[1] / "visualize.py"
        spec = importlib.util.spec_from_file_location("visualize", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_read_trial_csv_parses_metadata_headers_and_emg(self):
        module = self.load_script()

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "trial.csv"
            csv_path.write_text(SAMPLE_CSV, encoding="utf-8")

            trial = module.read_trial_csv(csv_path)

        self.assertEqual(trial.metadata["participant_id"], "yfy")
        self.assertEqual(trial.metadata["error_label"], "correct")
        self.assertEqual(trial.headers[0], "time_s")
        self.assertEqual(trial.emg.shape, (2, 2))

    def test_build_output_path_uses_input_folder_labels(self):
        module = self.load_script()

        path = module.build_output_path(
            csv_path=Path("root") / "yfy" / "high_clear" / "correct" / "trial_raw_emg.csv",
            input_dir=Path("root") / "yfy" / "high_clear" / "correct",
            output_dir=Path("visualize"),
        )

        self.assertEqual(
            path,
            Path("visualize") / "yfy_high_clear_correct" / "trial_raw_emg.png",
        )

    def test_visualize_folder_saves_one_png_per_csv(self):
        module = self.load_script()

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_dir = Path(tmp_dir) / "yfy" / "high_clear" / "correct"
            output_dir = Path(tmp_dir) / "visualize"
            input_dir.mkdir(parents=True)
            (input_dir / "trial_01_raw_emg.csv").write_text(SAMPLE_CSV, encoding="utf-8")
            (input_dir / "trial_02_raw_emg.csv").write_text(SAMPLE_CSV, encoding="utf-8")

            saved = module.visualize_folder(input_dir, output_dir)

            self.assertEqual(len(saved), 2)
            self.assertTrue(all(path.exists() for path in saved))
            self.assertTrue(all(path.suffix == ".png" for path in saved))


if __name__ == "__main__":
    unittest.main()
