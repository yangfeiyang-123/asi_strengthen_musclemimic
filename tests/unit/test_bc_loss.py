"""Tests for behavior cloning distillation losses."""

import json

import jax.numpy as jnp
import numpy as np
import distrax
import pytest

from musclemimic.distill.losses import bc_loss, distribution_mean, gaussian_diag_kl
from musclemimic.distill.eval_student import (
    run_eval_metrics,
    validate_required_metrics,
    write_comparison_outputs,
    write_summary_report,
)


def test_distribution_mean_supports_distrax_multivariate_normal_diag():
    pi = distrax.MultivariateNormalDiag(
        loc=jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        scale_diag=jnp.ones((2, 2)),
    )

    np.testing.assert_allclose(distribution_mean(pi), np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_bc_loss_uses_action_mse_and_optional_value_distill():
    student_mu = jnp.array([[0.0, 1.0], [2.0, 3.0]])
    teacher_action = jnp.array([[1.0, 1.0], [0.0, 3.0]])
    student_value = jnp.array([1.0, 4.0])
    teacher_value = jnp.array([3.0, 2.0])

    losses = bc_loss(
        student_mu=student_mu,
        teacher_action=teacher_action,
        student_value=student_value,
        teacher_value=teacher_value,
        action_mse_weight=1.0,
        value_distill_weight=0.5,
    )

    assert np.isclose(float(losses["action_mse"]), 1.25)
    assert np.isclose(float(losses["value_mse"]), 4.0)
    assert np.isclose(float(losses["total_loss"]), 3.25)


def test_bc_loss_without_teacher_value_has_zero_value_mse():
    losses = bc_loss(
        student_mu=jnp.zeros((2, 3)),
        teacher_action=jnp.ones((2, 3)),
        student_value=jnp.zeros(2),
        teacher_value=None,
    )

    assert np.isclose(float(losses["action_mse"]), 1.0)
    assert np.isclose(float(losses["value_mse"]), 0.0)
    assert np.isclose(float(losses["total_loss"]), 1.0)


def test_gaussian_diag_kl_matches_identical_zero_and_shifted_positive():
    teacher_mu = jnp.array([[0.0, 0.0]])
    teacher_log_std = jnp.array([[0.0, 0.0]])
    student_mu = jnp.array([[0.0, 0.0]])
    student_log_std = jnp.array([[0.0, 0.0]])

    assert np.isclose(float(gaussian_diag_kl(teacher_mu, teacher_log_std, student_mu, student_log_std)), 0.0)

    shifted = gaussian_diag_kl(
        teacher_mu,
        teacher_log_std,
        jnp.array([[1.0, 0.0]]),
        student_log_std,
    )
    assert np.isclose(float(shifted), 0.5)


def test_bc_loss_can_include_gaussian_kl():
    losses = bc_loss(
        student_mu=jnp.array([[1.0, 0.0]]),
        teacher_action=jnp.array([[0.0, 0.0]]),
        student_log_std=jnp.array([[0.0, 0.0]]),
        teacher_mu=jnp.array([[0.0, 0.0]]),
        teacher_log_std=jnp.array([[0.0, 0.0]]),
        gaussian_kl_weight=0.25,
    )

    assert np.isclose(float(losses["action_mse"]), 0.5)
    assert np.isclose(float(losses["gaussian_kl"]), 0.5)
    assert np.isclose(float(losses["total_loss"]), 0.625)


def test_comparison_outputs_include_dagger_policy(tmp_path):
    json_path, csv_path = write_comparison_outputs(
        {
            "teacher": {"val_mean_episode_return": 10.0},
            "student_bc_dagger": {"val_mean_episode_return": 7.5},
        },
        tmp_path,
    )

    assert json_path.is_file()
    text = csv_path.read_text(encoding="utf-8")
    assert "student_bc_dagger" in text


def test_summary_report_includes_acceptance_ratios(tmp_path):
    report_path = write_summary_report(
        {
            "teacher": {
                "mean_episode_return": 100.0,
                "completion_rate": 0.9,
                "early_termination_rate": 0.1,
            },
            "student_bc": {
                "mean_episode_return": 75.0,
                "completion_rate": 0.72,
                "early_termination_rate": 0.2,
            },
        },
        tmp_path,
    )

    text = report_path.read_text(encoding="utf-8")
    assert "student_bc" in text
    assert "return_ratio" in text
    assert "0.750000" in text


def test_validate_required_metrics_accepts_val_prefixed_metrics():
    validate_required_metrics(
        {
            "val_mean_episode_return": 1.0,
            "val_mean_episode_length": 10.0,
            "val_early_termination_rate": 0.0,
            "val_err_rpos": 0.1,
        }
    )


def test_validate_required_metrics_rejects_missing_metrics():
    with pytest.raises(RuntimeError, match="missing eval metrics"):
        validate_required_metrics({"mean_episode_return": 1.0})


def test_run_eval_metrics_prefers_json_metrics_output(monkeypatch):
    captured = {}

    def fake_run(cmd, check, text, capture_output):
        captured["cmd"] = cmd
        output_path = cmd[cmd.index("--metrics_output_json") + 1]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "mean_episode_return": 12.0,
                    "mean_episode_length": 34.0,
                    "early_termination_rate": 0.0,
                    "err_rpos": 0.25,
                },
                f,
            )

        class Result:
            stdout = "mean_episode_return: -1.0\n"

        return Result()

    monkeypatch.setattr("musclemimic.distill.eval_student.subprocess.run", fake_run)

    metrics = run_eval_metrics("/tmp/student", metrics_envs=2, metrics_steps=3, eval_seed=4)

    assert "--metrics_output_json" in captured["cmd"]
    assert metrics["mean_episode_return"] == 12.0
