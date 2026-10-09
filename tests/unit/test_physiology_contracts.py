"""Focused tests for effective excitation and IMR diagnostic numerics."""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from musclemimic.distill.physical import (
    MUSCLE_ACTIVATION_SEMANTICS as CANONICAL_ACTIVATION_SEMANTICS,
)
from musclemimic.distill.physical import (
    MUSCLE_EXCITATION_SEMANTICS as CANONICAL_EXCITATION_SEMANTICS,
)
from musclemimic.distill.physical import (
    physical_ctrl_to_effective_muscle_excitation,
    resolve_muscle_channel_contract,
)
from musclemimic.physiology.effective_excitation import (
    EFFECTIVE_EXCITATION_SEMANTICS,
    MUSCLE_ACTIVATION_SEMANTICS,
    actuator_transmission_target,
    effective_excitation_clip_diagnostics,
    effective_mujoco_muscle_excitation,
    jax_effective_muscle_excitation,
    normalized_policy_action_to_unit_muscle_ctrl,
    ordered_body_activation,
    resolve_muscle_channel_layout,
)
from musclemimic.physiology.intra_muscle import (
    IntraMuscleSpec,
    exact_exo_imr,
    robust_intra_muscle_consistency,
)
from environment.overall_environment.src.body_obs_adapter import _muscle_observations

jax.config.update("jax_platform_name", "cpu")


def _model_with_mixed_actuators() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(
        """
        <mujoco model="physiology_contract_test">
          <worldbody>
            <body name="body">
              <joint name="joint" type="hinge"/>
              <geom type="capsule" size=".02" fromto="0 0 0 0 0 .2" mass="1"/>
              <site name="origin" pos="0 0 .02"/>
              <site name="insertion" pos="0 0 .18"/>
            </body>
          </worldbody>
          <tendon>
            <spatial name="tendon_0"><site site="origin"/><site site="insertion"/></spatial>
            <spatial name="tendon_1"><site site="origin"/><site site="insertion"/></spatial>
            <spatial name="tendon_signed"><site site="origin"/><site site="insertion"/></spatial>
          </tendon>
          <actuator>
            <general name="muscle_0" tendon="tendon_0"
              ctrllimited="true" ctrlrange="0 1"
              dyntype="muscle" gaintype="muscle" biastype="muscle"
              dynprm=".01 .04" gainprm=".75 1.05 -1 200 .5 1.6 1.5 1.3 1.2 0"
              biasprm=".75 1.05 -1 200 .5 1.6 1.5 1.3 1.2 0"
              lengthrange=".05 .2"/>
            <motor name="motor" joint="joint" ctrlrange="-1 1"/>
            <general name="muscle_1" tendon="tendon_1"
              ctrllimited="true" ctrlrange="0 1"
              dyntype="muscle" gaintype="muscle" biastype="muscle"
              dynprm=".01 .04" gainprm=".75 1.05 -1 200 .5 1.6 1.5 1.3 1.2 0"
              biasprm=".75 1.05 -1 200 .5 1.6 1.5 1.3 1.2 0"
              lengthrange=".05 .2"/>
            <general name="muscle_signed" tendon="tendon_signed"
              ctrllimited="true" ctrlrange="-1 1"
              dyntype="muscle" gaintype="muscle" biastype="muscle"
              dynprm=".01 .04" gainprm=".75 1.05 -1 200 .5 1.6 1.5 1.3 1.2 0"
              biasprm=".75 1.05 -1 200 .5 1.6 1.5 1.3 1.2 0"
              lengthrange=".05 .2"/>
          </actuator>
        </mujoco>
        """
    )


def _spec(
    *,
    group_indices: list[list[int]],
    member_mask: list[list[float]],
    member_weights: list[list[float]] | None = None,
    deadband: list[float] | None = None,
    activity_off: list[float] | None = None,
    activity_on: list[float] | None = None,
    width: int = 4,
) -> IntraMuscleSpec:
    group_count = len(group_indices)
    if group_count == 0:
        group_indices_value = np.zeros((0, 1), dtype=np.int32)
        member_mask_value = np.zeros((0, 1), dtype=np.float32)
        member_weights_value = np.zeros((0, 1), dtype=np.float32)
    else:
        group_indices_value = group_indices
        member_mask_value = member_mask
        member_weights_value = member_mask if member_weights is None else member_weights
    return IntraMuscleSpec(
        group_indices=jnp.asarray(group_indices_value, dtype=jnp.int32),
        member_mask=jnp.asarray(member_mask_value, dtype=jnp.float32),
        member_weights=jnp.asarray(
            member_weights_value,
            dtype=jnp.float32,
        ),
        group_weights=jnp.ones((group_count,), dtype=jnp.float32),
        deadband=jnp.asarray(
            [0.1] * group_count if deadband is None else deadband,
            dtype=jnp.float32,
        ),
        activity_off=jnp.asarray(
            [0.0] * group_count if activity_off is None else activity_off,
            dtype=jnp.float32,
        ),
        activity_on=jnp.asarray(
            [0.1] * group_count if activity_on is None else activity_on,
            dtype=jnp.float32,
        ),
        activation_addresses=jnp.arange(width, dtype=jnp.int32),
        body_actuator_ids=jnp.arange(width, dtype=jnp.int32),
        group_ids=tuple(f"group_{index}" for index in range(group_count)),
    )


def test_effective_excitation_matches_mujoco_clamp_and_is_jittable():
    raw = jnp.asarray([-0.5, 0.0, 0.25, 1.0, 1.5], dtype=jnp.float32)
    effective = jax.jit(
        jax_effective_muscle_excitation,
        static_argnames=("backend",),
    )(
        raw,
        backend=jnp,
    )
    np.testing.assert_allclose(effective, [0.0, 0.0, 0.25, 1.0, 1.0])

    diagnostics = effective_excitation_clip_diagnostics(raw, backend=jnp)
    assert float(diagnostics["preclip_out_of_range_fraction"]) == pytest.approx(0.4)
    assert float(diagnostics["clip_correction_rms"]) > 0.0
    np.testing.assert_allclose(
        normalized_policy_action_to_unit_muscle_ctrl([-1.0, -0.8, 0.0, 0.8, 1.0]),
        [0.0, 0.1, 0.5, 0.9, 1.0],
    )

    model = _model_with_mixed_actuators()
    contract = resolve_muscle_channel_contract(model, ["muscle_1", "muscle_0"])
    assert effective_mujoco_muscle_excitation is physical_ctrl_to_effective_muscle_excitation
    np.testing.assert_allclose(
        effective_mujoco_muscle_excitation(
            [-0.5, 1.5],
            channel_contract=contract,
        ),
        [0.0, 1.0],
    )
    assert EFFECTIVE_EXCITATION_SEMANTICS == CANONICAL_EXCITATION_SEMANTICS
    assert MUSCLE_ACTIVATION_SEMANTICS == CANONICAL_ACTIVATION_SEMANTICS


def test_model_channel_layout_uses_actadr_and_rejects_nonmuscle_or_signed_contract():
    model = _model_with_mixed_actuators()
    layout = resolve_muscle_channel_layout(model, ["muscle_1", "muscle_0"])
    assert layout.actuator_ids.tolist() == [2, 0]
    assert layout.activation_addresses.tolist() == [1, 0]
    assert layout.activation_counts.tolist() == [1, 1]
    assert layout.actuator_schema_hash
    assert layout.runtime_model_hash
    assert actuator_transmission_target(model, 2)["name"] == "tendon_1"

    data = mujoco.MjData(model)
    data.act[:] = np.arange(model.na, dtype=np.float64) + 0.25
    np.testing.assert_allclose(
        ordered_body_activation(data, layout),
        data.act[[1, 0]],
    )
    with pytest.raises(ValueError, match="requires dyntype=muscle"):
        resolve_muscle_channel_layout(model, ["motor"])
    with pytest.raises(ValueError, match=r"ctrlrange.*exactly \[0,1\]"):
        resolve_muscle_channel_layout(model, ["muscle_signed"])
    signed = resolve_muscle_channel_layout(
        model,
        ["muscle_signed"],
        require_unit_ctrlrange=False,
    )
    np.testing.assert_array_equal(signed.ctrlrange, [[-1.0, 1.0]])


def test_body_observation_reads_packed_activation_address_not_actuator_id():
    model = _model_with_mixed_actuators()
    data = mujoco.MjData(model)
    data.ctrl[:] = [0.1, 0.2, 0.3, 0.4]
    data.act[:] = [0.25, 0.75, 0.5]
    mujoco.mj_forward(model, data)

    # muscle_1 has actuator id 2 but packed activation address 1.  Reading
    # data.act[actuator_id] would silently select the wrong muscle state.
    observations = _muscle_observations(
        model,
        data,
        ("muscle_1", "muscle_0"),
    )

    assert observations.shape == (10,)
    assert observations[4] == pytest.approx(data.act[1])
    assert observations[9] == pytest.approx(data.act[0])


def test_exact_exo_imr_matches_published_hard_deadband_sum():
    spec = _spec(
        group_indices=[[0, 1], [2, 3]],
        member_mask=[[1.0, 1.0], [1.0, 1.0]],
    )
    signal = jnp.asarray([0.0, 0.4, 0.6, 0.6], dtype=jnp.float32)
    metrics = exact_exo_imr(signal, spec)

    assert float(metrics.loss) == pytest.approx(0.08, abs=1e-7)
    assert metrics.group_loss.tolist() == pytest.approx([0.08, 0.0], abs=1e-7)
    assert float(metrics.violation_fraction) == pytest.approx(0.5)
    assert float(exact_exo_imr(signal, spec, deadband=0.2).loss) == 0.0


def test_robust_imr_uses_huber_deadband_activity_gate_and_group_normalization():
    spec = _spec(
        group_indices=[[0, 1], [2, 3]],
        member_mask=[[1.0, 1.0], [1.0, 1.0]],
        deadband=[0.1, 0.1],
        activity_off=[0.0, 0.7],
        activity_on=[0.1, 0.8],
    )
    signal = jnp.asarray([0.0, 0.4, 0.5, 0.9], dtype=jnp.float32)
    metrics = robust_intra_muscle_consistency(signal, spec, scale=0.1)

    assert metrics.group_loss.tolist() == pytest.approx([0.5, 0.5], abs=1e-6)
    assert metrics.group_activity_gate.tolist() == pytest.approx([1.0, 0.0])
    assert float(metrics.loss) == pytest.approx(0.5, abs=1e-6)
    assert float(metrics.active_group_fraction) == pytest.approx(0.5)
    assert float(metrics.violation_fraction) == pytest.approx(1.0)


def test_robust_imr_padding_empty_groups_vmap_and_grad_are_stable():
    padded = _spec(
        group_indices=[[0, 1, 0]],
        member_mask=[[1.0, 1.0, 0.0]],
        member_weights=[[1.0, 1.0, 999.0]],
    )
    signal = jnp.asarray([0.0, 0.4, 0.2, 0.2], dtype=jnp.float32)
    base_loss = robust_intra_muscle_consistency(signal, padded).loss
    compiled_loss = jax.jit(lambda value: robust_intra_muscle_consistency(value, padded).loss)(signal)
    assert float(compiled_loss) == pytest.approx(float(base_loss))
    gradient = jax.grad(lambda value: robust_intra_muscle_consistency(value, padded).loss)(signal)
    assert np.all(np.isfinite(np.asarray(gradient)))
    batched = jax.vmap(lambda value: robust_intra_muscle_consistency(value, padded).loss)(jnp.stack([signal, signal]))
    assert batched.tolist() == pytest.approx([float(base_loss), float(base_loss)])

    empty = _spec(
        group_indices=[],
        member_mask=[],
    )
    empty_metrics = jax.jit(lambda value: robust_intra_muscle_consistency(value, empty))(signal)
    assert float(empty_metrics.loss) == 0.0
    assert float(empty_metrics.active_group_fraction) == 0.0
    assert empty_metrics.group_loss.shape == (0,)
    assert float(exact_exo_imr(signal, empty).loss) == 0.0


def test_ordered_body_activation_from_static_spec_does_not_assume_actuator_id():
    spec = _spec(
        group_indices=[[0, 1]],
        member_mask=[[1.0, 1.0]],
    ).replace(
        activation_addresses=jnp.asarray([3, 1, 4, 0], dtype=jnp.int32),
        body_actuator_ids=jnp.asarray([8, 9, 10, 11], dtype=jnp.int32),
    )
    data = SimpleNamespace(act=jnp.asarray([0.1, 0.2, 0.3, 0.4, 0.5]))
    from musclemimic.physiology.intra_muscle import ordered_body_activation

    np.testing.assert_allclose(
        ordered_body_activation(data, spec),
        [0.4, 0.2, 0.5, 0.1],
    )
