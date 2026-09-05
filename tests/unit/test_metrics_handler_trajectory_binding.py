"""Regression tests for explicit MetricsHandler trajectory binding."""

from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import jax.numpy as jnp
import mujoco
import pytest
from omegaconf import OmegaConf

from musclemimic.utils.metrics import MetricsHandler

MODEL = mujoco.MjModel.from_xml_string(
    """
    <mujoco model="metrics_trajectory_binding">
      <worldbody>
        <body name="root">
          <joint name="root_joint" type="free"/>
          <geom size="0.1"/>
          <site name="pelvis_mimic"/>
        </body>
      </worldbody>
    </mujoco>
    """
)


class _Env:
    sites_for_mimic: ClassVar[list[str]] = ["pelvis_mimic"]

    def __init__(self, trajectory_data):
        self.th = (
            None
            if trajectory_data is None
            else SimpleNamespace(
                traj=SimpleNamespace(
                    data=trajectory_data,
                    info=SimpleNamespace(site_names=["pelvis_mimic"]),
                )
            )
        )

    def get_model(self):
        return MODEL

    def _get_all_info_properties(self):
        return {"root_free_joint_xml_name": "root_joint"}


def _config(*, measures):
    return OmegaConf.create(
        {
            "experiment": {
                "num_envs": 2,
                "validation": {
                    "measures": measures,
                    "quantities": [],
                    "rel_site_names": ["pelvis_mimic"],
                },
            }
        }
    )


def test_metrics_handler_sets_traj_data_from_handler():
    trajectory_data = SimpleNamespace(identity="bound")
    env = _Env(trajectory_data)
    handler = MetricsHandler(_config(measures=["EuclideanDistance"]), env)
    assert handler._traj_data is trajectory_data
    assert handler.requires_trajectory is True

    # Later handler mutation cannot make different metric paths read different
    # trajectory objects; construction occurs after sharing/JAX conversion.
    env.th.traj.data = SimpleNamespace(identity="replacement")
    assert handler._traj_data is trajectory_data


def test_metrics_handler_allows_none_when_no_measures_requested():
    handler = MetricsHandler(_config(measures=[]), _Env(None))
    assert handler._traj_data is None
    assert handler.requires_trajectory is False


def test_metrics_handler_rejects_measures_without_trajectory_data():
    with pytest.raises(ValueError, match="trajectory data is required"):
        MetricsHandler(_config(measures=["EuclideanDistance"]), _Env(None))


def test_metrics_handler_rejects_unknown_measure_with_value_error():
    with pytest.raises(ValueError, match="not a supported validation measure"):
        MetricsHandler(
            _config(measures=["not-a-measure"]),
            _Env(SimpleNamespace()),
        )


def test_frame_coverage_reads_the_same_bound_trajectory_data():
    original = SimpleNamespace(split_points=jnp.asarray([0, 10, 30]))
    env = _Env(original)
    handler = MetricsHandler(_config(measures=["EuclideanDistance"]), env)
    handler._root_qpos_ids = None
    env.th.traj.data = SimpleNamespace(split_points=jnp.asarray([0, 100, 200]))
    states = SimpleNamespace(
        metrics=SimpleNamespace(
            done=jnp.asarray([True, True]),
            returned_episode_returns=jnp.asarray([1.0, 1.0]),
            returned_episode_lengths=jnp.asarray([5.0, 5.0]),
            timestep=jnp.asarray([5, 5]),
            absorbing=jnp.asarray([False, False]),
        ),
        additional_carry=SimpleNamespace(
            traj_state=SimpleNamespace(
                traj_no=jnp.asarray([0, 1]),
                subtraj_step_no=jnp.asarray([4, 4]),
            )
        ),
        data=SimpleNamespace(),
    )

    summary = handler(states)

    assert float(summary.frame_coverage) == pytest.approx(10.0 / 30.0)
    assert handler._traj_data is original
