import sys
import traceback

import hydra

from musclemimic.utils.runtime_env import reexec_with_configured_cuda_env

reexec_with_configured_cuda_env()

import jax  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from musclemimic.runner.engine import run_experiment  # noqa: E402
from musclemimic.runner.logging import UnifiedHooks  # noqa: E402

jax.config.update("jax_default_matmul_precision", "high")


@hydra.main(version_base=None, config_path="./", config_name="conf_bimanual")
def experiment(config: DictConfig):
    try:
        run_experiment(config, hooks=UnifiedHooks())
    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    experiment()
