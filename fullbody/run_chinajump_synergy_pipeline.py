"""ChinaJump defaults for the transactional Stage-1 synergy pipeline.

This command prepares or validates artifacts only.  It never launches PPO,
initializes W&B, creates a checkpoint, or selects a GPU.  A successful
``training_ready_*`` release prints a shell bindings path that can subsequently
be sourced before the canonical ``scripts/run_fullbody_training.sh`` launcher.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence

from musclemimic.synergy.stage1_pipeline import (
    DEFAULT_BOOTSTRAP_CONFIG,
    DEFAULT_ENV_PREFIX,
    DEFAULT_FORMAL_CONFIG,
    DEFAULT_GROUPING,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PHASE_SCHEMA,
    DEFAULT_RESIDUAL_CONFIG,
)
from musclemimic.synergy.stage1_pipeline import (
    main as pipeline_main,
)

CHINAJUMP_DEFAULTS = {
    "target_skill_id": "ChinaJump",
    "env_prefix": DEFAULT_ENV_PREFIX,
    "grouping_json": DEFAULT_GROUPING,
    "phase_schema": DEFAULT_PHASE_SCHEMA,
    "output_root": DEFAULT_OUTPUT_ROOT,
    "formal_config_name": DEFAULT_FORMAL_CONFIG,
    "residual_config_name": DEFAULT_RESIDUAL_CONFIG,
    "bootstrap_config_name": DEFAULT_BOOTSTRAP_CONFIG,
}


def main(argv: Sequence[str] | None = None) -> int:
    return pipeline_main(argv, defaults=CHINAJUMP_DEFAULTS)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
