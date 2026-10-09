import importlib.util
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


class SixChannelScriptTest(unittest.TestCase):
    def load_script(self):
        script_path = Path(__file__).resolve().parents[1] / "delysis_measure_6ch.py"
        spec = importlib.util.spec_from_file_location("delysis_measure_6ch", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_uses_versioned_legacy_profile_without_imported_global_mutation(self):
        module = self.load_script()

        self.assertEqual(module.sensor_channels, [1, 2, 3, 4, 5, 6])
        self.assertEqual(module.LEGACY_HIGH_CLEAR_6CH.profile_id, "legacy_high_clear_6ch")
        source = (Path(__file__).resolve().parents[1] / "delysis_measure_6ch.py").read_text(encoding="utf-8")
        self.assertNotIn("dm.sensor_channels =", source)

    def test_sanitizes_participant_id_for_directory_names(self):
        module = self.load_script()

        self.assertEqual(module.sanitize_identifier(" player 01 "), "player_01")
        self.assertEqual(module.sanitize_identifier("Player-01"), "Player-01")
        with self.assertRaises(ValueError):
            module.sanitize_identifier("../bad")
        with self.assertRaises(ValueError):
            module.sanitize_identifier("   ")

    def test_build_trial_path_groups_by_participant_and_action(self):
        module = self.load_script()

        path = module.build_trial_path(
            base_dir=Path("root"),
            participant_id="player_01",
            action_label="high_clear",
            trial_index=3,
            timestamp="20260622_201500",
            error_label="correct",
        )

        self.assertEqual(
            path,
            Path("root")
            / "player_01"
            / "high_clear"
            / "correct"
            / "high_clear_20260622_201500_trial_03_correct_raw_emg.csv",
        )

    def test_normalizes_blank_error_label_as_correct(self):
        module = self.load_script()

        self.assertEqual(module.normalize_error_label(""), "correct")
        self.assertEqual(module.normalize_error_label("  "), "correct")
        self.assertEqual(module.normalize_error_label("elbow low"), "elbow_low")

    def test_save_trial_csv_writes_error_annotation_metadata(self):
        module = self.load_script()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "trial.csv"
            module.save_trial_csv(
                path=path,
                emg_arr=module.np.zeros((1, 6), dtype=module.np.float32),
                participant_id="player_01",
                action_label="high_clear",
                trial_index=1,
                timestamp="20260622_201500",
                error_label="elbow_low",
            )

            text = path.read_text(encoding="utf-8")

        self.assertIn("# movement_quality,error", text)
        self.assertIn("# error_label,elbow_low", text)

    def test_plot_trial_emg_creates_one_axis_per_selected_channel(self):
        module = self.load_script()
        emg_arr = module.np.zeros((20, 6), dtype=module.np.float32)

        fig = module.plot_trial_emg(
            emg_arr=emg_arr,
            participant_id="player_01",
            action_label="high_clear",
            trial_index=2,
            error_label="correct",
            show=False,
        )

        try:
            self.assertEqual(len(fig.axes), 6)
            self.assertTrue(all(axis.get_xlabel() == "Time (s)" for axis in fig.axes))
            self.assertTrue(all(axis.get_ylabel() == "EMG (mV)" for axis in fig.axes))
        finally:
            module.plt.close(fig)


if __name__ == "__main__":
    unittest.main()
