from __future__ import annotations

import pytest

from fullbody.run_distill_experiment import (
    DistillExperimentConfig,
    build_distill_experiment_plan,
)
from musclemimic.distill.collection_budget import resolve_collection_budget
from musclemimic.distill.dagger_loop import DaggerLoopConfig, build_iteration_plan


def test_total_transition_budget_trims_only_the_last_vector_batch():
    budget = resolve_collection_budget(
        num_envs=256,
        num_transitions=1_000_003,
        num_steps=None,
        default_transitions=1,
    )

    assert budget.requested_transitions == 1_000_003
    assert budget.vector_steps == 3907
    assert budget.planned_transitions_before_trim == 1_000_192
    assert budget.legacy_num_steps is None


def test_legacy_vector_steps_are_explicit_and_cannot_mix_with_transitions():
    legacy = resolve_collection_budget(
        num_envs=4,
        num_transitions=None,
        num_steps=3,
        default_transitions=100,
    )
    assert legacy.requested_transitions == 12
    assert legacy.legacy_num_steps == 3
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve_collection_budget(
            num_envs=4,
            num_transitions=10,
            num_steps=3,
            default_transitions=100,
        )


def test_production_direct_plan_uses_bounded_total_sample_budgets(tmp_path):
    plan = build_distill_experiment_plan(
        DistillExperimentConfig(
            teacher_ckpt="/teacher",
            test_only_allow_unpromoted_teacher=True,
            student_config="student.yaml",
            out_dir=str(tmp_path),
            train_motion_path=["train"],
            val_motion_path=["val"],
            collect_train=True,
            collect_val=True,
            train_bc=True,
            run_dagger=1,
        )
    )

    train = plan.commands["collect_train"]
    val = plan.commands["collect_val"]
    dagger = plan.commands["dagger"]
    assert train[train.index("--num_transitions") + 1] == "1000000"
    assert val[val.index("--num_transitions") + 1] == "200000"
    assert dagger[dagger.index("--num_transitions") + 1] == "500000"
    assert "--num_steps" not in train


def test_dagger_iteration_plan_uses_total_transitions_by_default(tmp_path):
    plan = build_iteration_plan(
        DaggerLoopConfig(
            teacher_ckpt="teacher",
            initial_student_ckpt="student",
            student_config="student.yaml",
            dataset_dir=str(tmp_path / "data"),
            output_dir=str(tmp_path / "out"),
            num_iters=1,
            test_only_allow_unpromoted_teacher=True,
        )
    )
    command = plan[0].collect_command
    assert command[command.index("--num_transitions") + 1] == "500000"
    assert "--num_steps" not in command
