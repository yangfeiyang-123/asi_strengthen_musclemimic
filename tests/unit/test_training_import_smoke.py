from __future__ import annotations

import subprocess
import sys


def test_training_distillation_and_latent_entrypoints_import_without_cycle():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import musclemimic.algorithms.ppo.runner; "
                "import musclemimic.distill.train_bc; "
                "import fullbody.latent_closed_loop_eval"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
