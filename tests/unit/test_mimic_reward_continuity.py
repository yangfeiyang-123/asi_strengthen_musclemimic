"""Online continuity reward plumbing and fail-closed mode tests."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

import musclemimic.core.reward.trajectory_based as reward_module
from musclemimic.algorithms.ppo.runner import _compute_training_metrics
from musclemimic.core.reward.trajectory_based import MimicReward
from musclemimic.core.wrappers.mjx import AutoResetWrapper
from musclemimic.core.wrappers.synergy_action import SynergyActionWrapper
from musclemimic.physiology.intra_muscle import FascicleContinuitySpec
from tests.unit.test_mimic_reward import (
    MINIMAL_MJCF,
    FakeTrajectoryHandler,
    make_carry,
    make_env,
    make_sim_data,
    make_traj_data,
)
from tests.unit.test_mjx_reset import _AutoResetSmokeEnv
from tests.unit.test_synergy_action_wrapper import (
    _artifacts,
    _config,
    _MockBodyEnv,
    _MockState,
)


def _continuity_spec() -> FascicleContinuitySpec:
    return FascicleContinuitySpec(
        edge_indices=jnp.asarray([[[0, 1]]], dtype=jnp.int32),
        edge_mask=jnp.asarray([[1.0]], dtype=jnp.float32),
        edge_weights=jnp.asarray([[1.0]], dtype=jnp.float32),
        member_indices=jnp.asarray([[0, 1]], dtype=jnp.int32),
        member_mask=jnp.asarray([[1.0, 1.0]], dtype=jnp.float32),
        member_weights=jnp.asarray([[1.0, 1.0]], dtype=jnp.float32),
        chain_weights=jnp.asarray([1.0], dtype=jnp.float32),
        deadband=jnp.asarray([0.1], dtype=jnp.float32),
        activity_off=jnp.asarray([0.0], dtype=jnp.float32),
        activity_on=jnp.asarray([0.1], dtype=jnp.float32),
        activation_addresses=jnp.asarray([0, 1], dtype=jnp.int32),
        body_actuator_ids=jnp.asarray([0, 1], dtype=jnp.int32),
        chain_ids=("fixture_chain",),
    )


def _minimal_reward_fixture():
    model = mujoco.MjModel.from_xml_string(MINIMAL_MJCF)
    qpos = np.asarray([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    handler = FakeTrajectoryHandler(make_traj_data(qpos, backend=np))
    env = make_env(model, handler)
    carry = make_carry()
    data = make_sim_data(qpos, backend=np)
    data.act = np.asarray([0.2, 0.8, 0.0])
    return model, env, carry, data


def _evaluate(reward: MimicReward, model, env, carry, data):
    return reward(
        state=np.zeros(10),
        action=np.zeros(3),
        next_state=np.zeros(10),
        absorbing=False,
        info={},
        env=env,
        model=model,
        data=data,
        carry=carry,
        backend=np,
    )


def test_diagnostics_only_is_reward_and_carry_equivalent_to_off():
    model, env, carry, data = _minimal_reward_fixture()
    disabled = MimicReward(env)
    diagnostic = copy.copy(disabled)
    diagnostic._fascicle_continuity_mode = "diagnostics"
    diagnostic._fascicle_continuity_compute = True
    diagnostic._fascicle_continuity_reward_active = False
    diagnostic._fascicle_continuity_spec = _continuity_spec()
    diagnostic._fascicle_continuity_measured_chain_count = 1
    diagnostic._fascicle_continuity_measured_edge_count = 1

    off_reward, off_carry, off_info = _evaluate(disabled, model, env, carry, data)
    diagnostic_reward, diagnostic_carry, diagnostic_info = _evaluate(
        diagnostic,
        model,
        env,
        carry,
        data,
    )

    assert diagnostic_reward == pytest.approx(off_reward, abs=0.0)
    np.testing.assert_array_equal(
        diagnostic_carry.reward_state.last_qvel,
        off_carry.reward_state.last_qvel,
    )
    np.testing.assert_array_equal(
        diagnostic_carry.reward_state.last_action,
        off_carry.reward_state.last_action,
    )
    assert diagnostic_info["penalty_fascicle_continuity"] == 0.0
    assert diagnostic_info["fascicle_continuity_loss"] > 0.0
    assert diagnostic_info["fascicle_continuity_measured_chain_count"] == 1
    assert diagnostic_info["fascicle_continuity_measured_edge_count"] == 1
    assert off_info["fascicle_continuity_measured_chain_count"] == 0


def test_reward_mode_logs_raw_weighted_and_total_penalty_before_clip():
    model, env, carry, data = _minimal_reward_fixture()
    reward = MimicReward(env)
    reward._fascicle_continuity_mode = "reward"
    reward._fascicle_continuity_compute = True
    reward._fascicle_continuity_reward_active = True
    reward._fascicle_continuity_spec = _continuity_spec()
    reward._fascicle_continuity_reward_spec = _continuity_spec()
    reward._fascicle_continuity_coefficient = 10.0
    reward._fascicle_continuity_raw_penalty_clip = None
    reward._fascicle_continuity_measured_chain_count = 1
    reward._fascicle_continuity_measured_edge_count = 1

    _, _, info = _evaluate(reward, model, env, carry, data)

    assert info["fascicle_continuity_training_loss"] == pytest.approx(info["fascicle_continuity_loss"])
    assert info["penalty_fascicle_continuity"] == pytest.approx(-10.0 * info["fascicle_continuity_training_loss"])
    assert info["penalty_total_before_clip"] < -1.0
    assert info["penalty_total"] == pytest.approx(-1.0)


@pytest.mark.parametrize(
    "config",
    [
        {"mode": "diagnostics", "coefficient": 0.1},
        {"mode": "off", "coefficient": 0.1},
        {"mode": "reward", "coefficient": 0.0},
        {"mode": "reward", "coefficient": -0.1},
    ],
)
def test_mode_and_coefficient_contract_is_fail_closed(config):
    reward = object.__new__(MimicReward)
    with pytest.raises(ValueError, match="coefficient"):
        reward._configure_fascicle_continuity(SimpleNamespace(), config)


def _mock_online_binding(
    monkeypatch,
    tmp_path,
    *,
    reward_gate_error: str | None = None,
    promotion: dict | None = None,
):
    taxonomy_path = tmp_path / "taxonomy.json"
    graph_path = tmp_path / "graph.json"
    taxonomy_path.touch()
    graph_path.touch()
    names = tuple(f"muscle_{index:03d}" for index in range(354))
    taxonomy = SimpleNamespace(
        model_binding={
            "target": {
                "environment": "MyoFullBody",
                "disable_fingers": True,
                "expected_action_dim": 354,
            }
        },
        actuator_names=names,
        fingerprint="a" * 64,
    )
    graph = SimpleNamespace(
        taxonomy_binding={"runtime_compatibility": "portable_muscle_channel_abi"},
        graph_fingerprint="b" * 64,
        graph_id="fixture_graph",
        training_enabled_chain_count=int(promotion is not None),
        generation=None if promotion is None else {"training_promotion": promotion},
    )
    monkeypatch.setattr(reward_module, "load_anatomical_taxonomy", lambda _path: taxonomy)
    monkeypatch.setattr(reward_module, "validate_taxonomy_against_model", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        reward_module,
        "resolve_ordered_policy_muscle_layout",
        lambda *args, **kwargs: SimpleNamespace(actuator_names=names),
    )
    monkeypatch.setattr(
        reward_module,
        "load_fascicle_continuity_graph",
        lambda *args, **kwargs: graph,
    )
    monkeypatch.setattr(
        reward_module,
        "validate_continuity_graph_against_model",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        reward_module,
        "build_fascicle_continuity_spec",
        lambda *args, **kwargs: _continuity_spec(),
    )

    def gate(*args, **kwargs):
        if reward_gate_error is not None:
            raise ValueError(reward_gate_error)
        return False, "diagnostics"

    monkeypatch.setattr(reward_module, "resolve_fascicle_continuity_reward_gate", gate)
    return taxonomy_path, graph_path


def test_diagnostics_initialization_binds_graph_taxonomy_policy_order_and_coverage(
    monkeypatch,
    tmp_path,
):
    taxonomy_path, graph_path = _mock_online_binding(monkeypatch, tmp_path)
    reward = object.__new__(MimicReward)
    reward._configure_fascicle_continuity(
        SimpleNamespace(_model=object()),
        {
            "mode": "diagnostics",
            "coefficient": 0.0,
            "taxonomy_path": str(taxonomy_path),
            "continuity_path": str(graph_path),
            "expected_taxonomy_fingerprint": "a" * 64,
            "expected_continuity_fingerprint": "b" * 64,
        },
    )

    assert reward._fascicle_continuity_compute is True
    assert reward._fascicle_continuity_reward_active is False
    assert reward._fascicle_continuity_measured_chain_count == 1
    assert reward._fascicle_continuity_measured_edge_count == 1


def test_reward_initialization_rejects_graph_without_verified_training_chains(
    monkeypatch,
    tmp_path,
):
    taxonomy_path, graph_path = _mock_online_binding(
        monkeypatch,
        tmp_path,
        reward_gate_error="no verified training-enabled chains",
    )
    reward = object.__new__(MimicReward)
    with pytest.raises(ValueError, match="no verified training-enabled chains"):
        reward._configure_fascicle_continuity(
            SimpleNamespace(_model=object()),
            {
                "mode": "reward",
                "coefficient": 0.01,
                "taxonomy_path": str(taxonomy_path),
                "continuity_path": str(graph_path),
                "expected_taxonomy_fingerprint": "a" * 64,
                "expected_continuity_fingerprint": "b" * 64,
            },
        )


@pytest.mark.parametrize(
    ("promotion", "config_overrides", "error_match"),
    [
        (
            {
                "calibration_fingerprint": "c" * 64,
                "selected_reward_coefficient": 0.01,
            },
            {},
            "pinned calibration fingerprint",
        ),
        (None, {"expected_calibration_fingerprint": "c" * 64}, "lacks training-promotion"),
        (
            {
                "calibration_fingerprint": "d" * 64,
                "selected_reward_coefficient": 0.01,
            },
            {"expected_calibration_fingerprint": "c" * 64},
            "calibration differs",
        ),
        (
            {"calibration_fingerprint": "c" * 64},
            {"expected_calibration_fingerprint": "c" * 64},
            "lacks a calibrated coefficient",
        ),
        (
            {
                "calibration_fingerprint": "c" * 64,
                "selected_reward_coefficient": 0.02,
            },
            {"expected_calibration_fingerprint": "c" * 64},
            "coefficient differs",
        ),
    ],
)
def test_reward_initialization_requires_exact_calibration_binding(
    monkeypatch,
    tmp_path,
    promotion,
    config_overrides,
    error_match,
):
    taxonomy_path, graph_path = _mock_online_binding(
        monkeypatch,
        tmp_path,
        promotion=promotion,
    )
    config = {
        "mode": "reward",
        "coefficient": 0.01,
        "taxonomy_path": str(taxonomy_path),
        "continuity_path": str(graph_path),
        "expected_taxonomy_fingerprint": "a" * 64,
        "expected_continuity_fingerprint": "b" * 64,
        **config_overrides,
    }
    reward = object.__new__(MimicReward)
    with pytest.raises(ValueError, match=error_match):
        reward._configure_fascicle_continuity(SimpleNamespace(_model=object()), config)


def test_reward_initialization_accepts_exact_calibration_and_coefficient_binding(
    monkeypatch,
    tmp_path,
):
    calibration_fingerprint = "c" * 64
    taxonomy_path, graph_path = _mock_online_binding(
        monkeypatch,
        tmp_path,
        promotion={
            "calibration_fingerprint": calibration_fingerprint,
            "selected_reward_coefficient": 0.01,
        },
    )
    reward = object.__new__(MimicReward)
    reward._configure_fascicle_continuity(
        SimpleNamespace(_model=object()),
        {
            "mode": "reward",
            "coefficient": 0.01,
            "taxonomy_path": str(taxonomy_path),
            "continuity_path": str(graph_path),
            "expected_taxonomy_fingerprint": "a" * 64,
            "expected_continuity_fingerprint": "b" * 64,
            "expected_calibration_fingerprint": calibration_fingerprint,
        },
    )

    assert reward._fascicle_continuity_reward_active is True
    assert reward._fascicle_continuity_reward_spec is not None


@pytest.mark.parametrize("residual", [False, True])
def test_fixed_synergy_and_residual_wrappers_preserve_continuity_info(tmp_path, residual):
    basis, stats, residual_basis = _artifacts(tmp_path, residual=residual)
    base = _MockBodyEnv()

    def step(state, action):
        base.last_body_action = action
        return (
            jnp.ones(4),
            1.0,
            False,
            False,
            {
                "fascicle_continuity_loss": jnp.asarray(0.25),
                "fascicle_continuity_measured_edge_count": jnp.asarray(140.0),
            },
            state,
        )

    base.step = step
    wrapper = SynergyActionWrapper(base, _config(basis, stats, residual_basis))
    result = wrapper.step(
        _MockState(step=jnp.asarray(0), info={}),
        jnp.zeros(wrapper.info.action_space.shape[0]),
    )

    assert float(result[4]["fascicle_continuity_loss"]) == pytest.approx(0.25)
    assert float(result[4]["fascicle_continuity_measured_edge_count"]) == 140.0


def test_autoreset_transition_state_preserves_terminal_continuity_metrics():
    base = _AutoResetSmokeEnv()
    original_step = base.step

    def step(state, action):
        result = list(original_step(state, action))
        next_state = result[-1].replace(
            info={
                "fascicle_continuity_loss": jnp.asarray([0.75, 0.25]),
                "fascicle_continuity_measured_edge_count": jnp.asarray([140.0, 140.0]),
            }
        )
        result[4] = next_state.info
        result[-1] = next_state
        return tuple(result)

    base.step = step
    env = AutoResetWrapper(base)
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    _, state = env.reset(keys)

    _, _, _, done, info, _, transition_state = env.step_with_transition(
        state,
        jnp.zeros((2, 1), dtype=jnp.float32),
    )

    np.testing.assert_array_equal(np.asarray(done), [True, False])
    np.testing.assert_allclose(
        np.asarray(transition_state.info["fascicle_continuity_loss"]),
        [0.75, 0.25],
    )
    np.testing.assert_allclose(
        np.asarray(info["fascicle_continuity_measured_edge_count"]),
        [140.0, 140.0],
    )


def test_ppo_summary_keeps_continuity_coverage_and_preclip_penalty():
    zeros = jnp.zeros((2,), dtype=jnp.float32)
    info = dict.fromkeys(
        (
            "reward_qpos",
            "reward_qvel",
            "reward_root_pos",
            "reward_rpos",
            "reward_rquat",
            "reward_rvel_rot",
            "reward_rvel_lin",
            "reward_root_vel",
            "penalty_action_saturation",
            "penalty_activation_energy",
            "err_root_xyz",
            "err_root_yaw",
            "err_joint_pos",
            "err_joint_vel",
            "err_site_abs",
            "err_rpos",
        ),
        zeros,
    )
    info.update(
        {
            "reward_total": jnp.asarray([1.0, 0.5]),
            "penalty_total": jnp.asarray([-1.0, -0.5]),
            "penalty_total_before_clip": jnp.asarray([-3.0, -0.5]),
            "penalty_fascicle_continuity": jnp.asarray([-2.0, 0.0]),
            "fascicle_continuity_loss": jnp.asarray([0.4, 0.2]),
            "fascicle_continuity_measured_chain_count": jnp.asarray([28.0, 28.0]),
            "fascicle_continuity_measured_edge_count": jnp.asarray([140.0, 140.0]),
        }
    )
    metrics = SimpleNamespace(
        done=jnp.asarray([True, False]),
        returned_episode_returns=jnp.asarray([1.0, 0.0]),
        returned_episode_lengths=jnp.asarray([10.0, 0.0]),
        timestep=jnp.asarray([10, 10]),
    )
    batch = SimpleNamespace(
        metrics=metrics,
        absorbing=jnp.asarray([False, False]),
        info=info,
    )

    summary = _compute_training_metrics(batch, SimpleNamespace(num_envs=2))

    assert float(summary.penalty_total_before_clip) == pytest.approx(-1.75)
    assert float(summary.penalty_fascicle_continuity) == pytest.approx(-1.0)
    assert float(summary.fascicle_continuity_loss) == pytest.approx(0.3)
    assert float(summary.fascicle_continuity_measured_chain_count) == 28.0
    assert float(summary.fascicle_continuity_measured_edge_count) == 140.0
