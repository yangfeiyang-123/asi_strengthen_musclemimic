#!/usr/bin/env python3
"""Produce sealed Stage-1 PEASD endpoint validation evidence.

T1--T4 are normally sealed automatically by the PPO runner.  This command is
also the canonical post-training producer for T0: it loads the fixed-budget
endpoint, attaches a verified EMG tube in reward-neutral ``diagnostics_only``
mode, evaluates every held-out trajectory exactly once, and seals the result
against the immutable checkpoint/history/manifest contracts.

This command evaluates an existing checkpoint only.  It never launches or
resumes training.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf, open_dict

from musclemimic.utils.runtime_env import reexec_with_configured_cuda_env

reexec_with_configured_cuda_env()


def _t0_diagnostics_config(
    experiment: Any,
    *,
    reference_cache: str | Path,
) -> dict[str, Any]:
    budget = experiment.get("stage1_peasd_fixed_budget_contract", {})
    tube_action_id = str(budget.get("tube_action_id", "") or "").strip()
    if not tube_action_id:
        raise ValueError("T0 fixed-budget contract has no primary tube action identity")
    reference_path = Path(reference_cache).expanduser().resolve(strict=True)
    raw = experiment.get("env_params", {}).get("reward_params", {}).get(
        "emg_consistency", {}
    )
    base = (
        OmegaConf.to_container(raw, resolve=True)
        if OmegaConf.is_config(raw)
        else dict(raw)
    )
    base.update(
        {
            "enabled": True,
            "mode": "diagnostics_only",
            "arm": "T0",
            "action_id": tube_action_id,
            "reference_cache": str(reference_path),
            "mapping_path": None,
            "anchor_weight_max": 0.0,
            "synergy_weight_max": 0.0,
            "synergy_phase_shuffle_offset_bins": 0,
            "use_activation": True,
        }
    )
    return base


def evaluate_stage1_peasd_checkpoint(
    checkpoint: str | Path,
    *,
    reference_cache: str | Path | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    """Run one strict endpoint evaluation and write sealed evidence.

    ``reference_cache`` is required only for T0.  Active arms recompile and
    compare the exact EMG runtime already frozen in the training manifest.
    """

    from musclemimic.algorithms import PPOJax
    from musclemimic.algorithms.common.env_utils import wrap_env
    from musclemimic.algorithms.ppo.runner import _run_strict_promotion_validation
    from musclemimic.badminton.promotion_artifact import checkpoint_identity
    from musclemimic.physiology.emg_consistency_runtime import (
        build_emg_consistency_preflight_contract,
    )
    from musclemimic.runner.engine import instantiate_validation_env
    from musclemimic.runner.eval_utils import (
        align_agent_state,
        load_checkpoint,
        resolve_checkpoint_path,
    )
    from musclemimic.runner.stage1_peasd_validation import (
        build_stage1_peasd_validation_evidence,
        validate_stage1_peasd_validation_history,
        write_stage1_peasd_validation_evidence,
    )

    checkpoint_path = Path(resolve_checkpoint_path(str(checkpoint))).expanduser().resolve(strict=True)
    identity = checkpoint_identity(checkpoint_path)
    config, agent_state, _metadata = load_checkpoint(str(checkpoint_path))
    OmegaConf.set_struct(config, False)
    experiment = config.experiment
    stage1 = experiment.get("stage1_peasd", {})
    arm = str(stage1.get("arm", "") or "").strip().upper()
    if arm not in {"T0", "T1", "T2", "T3", "T4"}:
        raise ValueError("checkpoint is not a Stage1 PEASD T0--T4 run")
    budget = experiment.get("stage1_peasd_fixed_budget_contract", {})
    if (
        int(identity.get("update_number", -1))
        != int(budget.get("expected_endpoint_update_number", -2))
        or int(identity.get("global_timestep", -1))
        != int(budget.get("expected_endpoint_global_timestep", -2))
    ):
        raise ValueError("Stage1 PEASD evidence may only evaluate the fixed-budget endpoint")

    versioned_history = (
        checkpoint_path.parent
        / "stage1_peasd_validation_history"
        / f"checkpoint_{int(identity['update_number'])}.json"
    ).resolve(strict=True)
    history = validate_stage1_peasd_validation_history(
        versioned_history,
        require_complete=True,
    )
    provenance = dict(history["validation_provenance"])
    eval_seed = int(provenance["eval_seed"])

    if arm == "T0":
        if reference_cache is None:
            raise ValueError("T0 post-hoc physiology evaluation requires --reference-cache")
        diagnostic_config = _t0_diagnostics_config(
            experiment,
            reference_cache=reference_cache,
        )
        # Fail on release/mapping/trial-QC before constructing a model or
        # compiling JAX.  This preflight is diagnostics-only and is never
        # written back into the immutable T0 training manifest.
        build_emg_consistency_preflight_contract(
            diagnostic_config,
            base_dir=Path(__file__).resolve().parents[1],
        )
        with open_dict(experiment.env_params.reward_params):
            experiment.env_params.reward_params.emg_consistency = diagnostic_config
    elif reference_cache is not None:
        raise ValueError("--reference-cache is reserved for the tube-free T0 checkpoint")

    validation = experiment.get("validation", {})
    if (
        not bool(validation.get("active", False))
        or not bool(validation.get("deterministic", False))
        or not bool(validation.get("start_from_beginning", False))
    ):
        raise ValueError("Stage1 checkpoint lacks its deterministic frame-zero validation contract")

    env = instantiate_validation_env(config, share_trajectory=False)
    if env is None or getattr(env, "th", None) is None:
        raise ValueError("Stage1 post-hoc validation could not load the held-out trajectory split")
    runtime_contract = env._reward_function.emg_consistency_runtime_contract
    if runtime_contract is None:
        raise ValueError("Stage1 physiology evaluation did not compile an EMG runtime")
    if arm != "T0":
        training_runtime = experiment.get("emg_consistency_runtime_contract", None)
        plain_training_runtime = (
            OmegaConf.to_container(training_runtime, resolve=True)
            if OmegaConf.is_config(training_runtime)
            else dict(training_runtime or {})
        )
        if plain_training_runtime != runtime_contract:
            raise ValueError("active-arm evaluation runtime differs from the training manifest")

    agent_conf = PPOJax.init_agent_conf(env, config)
    agent_state = align_agent_state(agent_state, agent_conf)
    wrapped_validation_env = wrap_env(env, experiment)
    metrics = _run_strict_promotion_validation(
        network=agent_conf.network,
        train_state=agent_state.train_state,
        val_env=wrapped_validation_env,
        config=experiment,
        eval_seed=eval_seed,
    )
    evidence = build_stage1_peasd_validation_evidence(
        checkpoint_identity_payload=identity,
        validation_provenance=provenance,
        metrics=metrics,
        validation_history=versioned_history,
        evaluation_runtime_contract=runtime_contract,
    )
    versioned_evidence, latest_evidence = write_stage1_peasd_validation_evidence(evidence)
    return evidence, versioned_evidence, latest_evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Exact Stage1 endpoint checkpoint leaf")
    parser.add_argument(
        "--reference-cache",
        default=None,
        help="Verified tube directory/file; required for T0 diagnostics-only evaluation",
    )
    args = parser.parse_args()
    evidence, versioned, latest = evaluate_stage1_peasd_checkpoint(
        args.checkpoint,
        reference_cache=args.reference_cache,
    )
    print(f"Stage1 PEASD validation evidence: {versioned}")
    print(f"Stage1 PEASD validation latest: {latest}")
    print(f"Stage1 PEASD evidence fingerprint: {evidence['evidence_fingerprint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
