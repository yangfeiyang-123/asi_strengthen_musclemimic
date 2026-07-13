import sys

import pytest

from musclemimic.badminton.scripts import (
    collect_forehand_clear_dagger_dataset as dagger_wrapper,
)
from musclemimic.badminton.scripts import (
    collect_forehand_clear_teacher_dataset as teacher_wrapper,
)
from musclemimic.badminton.scripts import (
    evaluate_forehand_clear_student as evaluate_wrapper,
)
from musclemimic.badminton.scripts import (
    run_forehand_clear_dagger_loop as loop_wrapper,
)


@pytest.mark.parametrize(
    ("module", "required_args", "expected"),
    [
        (
            teacher_wrapper,
            [
                "--teacher-path",
                "teacher",
                "--output-dir",
                "dataset",
                "--test-only-allow-unpromoted-teacher",
            ],
            "1000000",
        ),
        (
            dagger_wrapper,
            [
                "--teacher-path",
                "teacher",
                "--student-path",
                "student",
                "--output-dir",
                "dataset",
                "--dagger-iteration",
                "0",
                "--resume-dataset",
                "--test-only-allow-unpromoted-teacher",
            ],
            "500000",
        ),
        (
            loop_wrapper,
            [
                "--teacher-path",
                "teacher",
                "--student-path",
                "student",
                "--dataset-dir",
                "dataset",
                "--output-dir",
                "loop",
                "--test-only-allow-unpromoted-teacher",
            ],
            "500000",
        ),
    ],
)
def test_forehand_wrappers_default_to_total_transition_budgets(
    monkeypatch, module, required_args, expected
):
    commands = []
    monkeypatch.setattr(sys, "argv", ["wrapper", *required_args])
    monkeypatch.setattr(module.subprocess, "run", lambda command, **_: commands.append(command))

    assert module.main() == 0

    command = commands[0]
    assert "--num_transitions" in command
    assert command[command.index("--num_transitions") + 1] == expected
    assert "--num_steps" not in command


def test_teacher_wrapper_keeps_legacy_vector_steps_explicit(monkeypatch):
    commands = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wrapper",
            "--teacher-path",
            "teacher",
            "--output-dir",
            "dataset",
            "--num-steps",
            "7",
            "--test-only-allow-unpromoted-teacher",
        ],
    )
    monkeypatch.setattr(
        teacher_wrapper.subprocess,
        "run",
        lambda command, **_: commands.append(command),
    )

    assert teacher_wrapper.main() == 0

    command = commands[0]
    assert command[command.index("--num_steps") + 1] == "7"
    assert "--num_transitions" not in command


def test_teacher_wrapper_forwards_resume_run_and_promotion(monkeypatch):
    commands = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wrapper",
            "--teacher-path",
            "teacher",
            "--teacher-promotion-manifest",
            "stage2.json",
            "--output-dir",
            "dataset",
            "--resume-dataset",
            "--run-uid",
            "run-123",
        ],
    )
    monkeypatch.setattr(
        teacher_wrapper.subprocess,
        "run",
        lambda command, **_: commands.append(command),
    )

    assert teacher_wrapper.main() == 0
    command = commands[0]
    assert command[command.index("--teacher-promotion-manifest") + 1] == "stage2.json"
    assert "--resume-dataset" in command
    assert command[command.index("--run-uid") + 1] == "run-123"


def test_dagger_wrapper_requires_and_forwards_iteration_resume_promotion(monkeypatch):
    commands = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wrapper",
            "--teacher-path",
            "teacher",
            "--student-path",
            "student",
            "--teacher-promotion-manifest",
            "stage2.json",
            "--output-dir",
            "dataset",
            "--dagger-iteration",
            "2",
            "--resume-dataset",
            "--run-uid",
            "run-123",
        ],
    )
    monkeypatch.setattr(
        dagger_wrapper.subprocess,
        "run",
        lambda command, **_: commands.append(command),
    )

    assert dagger_wrapper.main() == 0
    command = commands[0]
    assert command[command.index("--dagger-iteration") + 1] == "2"
    assert "--resume-dataset" in command
    assert command[command.index("--run-uid") + 1] == "run-123"
    assert command[command.index("--teacher-promotion-manifest") + 1] == "stage2.json"
    assert "--append" not in command


def test_evaluation_wrapper_forwards_production_acceptance_evidence(monkeypatch):
    commands = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "wrapper",
            "--teacher-path",
            "teacher",
            "--student-path",
            "student",
            "--output-dir",
            "comparison",
            "--dataset-dir",
            "dataset",
            "--convergence-metrics",
            "convergence.json",
            "--motion-path",
            "heldout-1",
            "--require-pass",
        ],
    )
    monkeypatch.setattr(
        evaluate_wrapper.subprocess,
        "run",
        lambda command, **_: commands.append(command),
    )

    assert evaluate_wrapper.main() == 0
    command = commands[0]
    assert command[command.index("--dataset_dir") + 1] == "dataset"
    assert command[command.index("--convergence_metrics") + 1] == (
        "convergence.json"
    )
    assert "--require_pass" in command
