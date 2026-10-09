import sys
import traceback
from datetime import datetime
from pathlib import Path

import hydra
from musclemimic.utils.runtime_env import reexec_with_configured_cuda_env

reexec_with_configured_cuda_env()

import jax
from omegaconf import DictConfig

from musclemimic.runner.engine import run_experiment
from musclemimic.runner.logging import UnifiedHooks

jax.config.update("jax_default_matmul_precision", "high")


def _override_value(argv: list[str], key: str) -> str | None:
    prefixes = (f"{key}=", f"+{key}=")
    for arg in argv[1:]:
        if arg.startswith(prefixes):
            return arg.split("=", 1)[1].strip("'\"")
    return None


def _append_action_hydra_run_dir(argv: list[str]) -> None:
    if any(arg.startswith(("hydra.run.dir=", "+hydra.run.dir=")) for arg in argv[1:]):
        return
    training_root = _override_value(argv, "experiment.training_root")
    if training_root:
        root = Path(training_root)
        if not root.is_absolute():
            root = Path.cwd() / root
    else:
        action = _override_value(argv, "experiment.training_action")
        if not action:
            return
        root = Path.cwd() / "datasets" / action / "training"
    run_id = datetime.now().strftime("%Y-%m-%d/%H-%M-%S")
    argv.append(f"hydra.run.dir={root / 'hydra' / run_id}")


@hydra.main(version_base=None, config_path="./", config_name="conf_fullbody")
def experiment(config: DictConfig):
    try:
        run_experiment(config, hooks=UnifiedHooks())
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    _append_action_hydra_run_dir(sys.argv)
    experiment()
