"""Exact no-finger MyoFullBody muscle and continuity asset contract."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from musclemimic.environments.humanoids.myofullbody import MyoFullBody
from musclemimic.physiology.anatomical_groups import (
    load_anatomical_taxonomy,
    validate_taxonomy_against_model,
)
from musclemimic.physiology.continuity_groups import (
    load_fascicle_continuity_graph,
    validate_continuity_graph_against_model,
)
from musclemimic.physiology.runtime_binding import (
    ordered_policy_actuator_names,
    resolve_muscle_activation_addresses,
    resolve_ordered_policy_muscle_layout,
)

ROOT = Path(__file__).resolve().parents[2]


def test_myofullbody_no_finger_exact_354_contract_and_graph_binding():
    env = MyoFullBody(disable_fingers=True)
    model = env._model
    taxonomy = load_anatomical_taxonomy(ROOT / "configs/physiology/myofullbody_354_muscle_taxonomy_curated_v1.json")
    graph = load_fascicle_continuity_graph(
        ROOT / "configs/physiology/myofullbody_354_fascicle_continuity_v1.json",
        taxonomy=taxonomy,
    )

    layout = resolve_ordered_policy_muscle_layout(env)
    assert len(ordered_policy_actuator_names(env)) == 354
    assert layout.actuator_names == taxonomy.actuator_names
    assert layout.actuator_ids.shape == (354,)
    assert layout.activation_addresses.shape == (354,)
    assert np.unique(np.asarray(layout.activation_addresses)).size == 354
    np.testing.assert_allclose(layout.ctrlrange, np.asarray([[0.0, 1.0]] * 354))
    np.testing.assert_array_equal(
        resolve_muscle_activation_addresses(model),
        np.asarray(layout.activation_addresses),
    )
    assert all(
        int(model.actuator_dyntype[index]) == int(mujoco.mjtDyn.mjDYN_MUSCLE)
        for index in np.asarray(layout.actuator_ids).tolist()
    )

    validate_taxonomy_against_model(taxonomy, model)
    validate_continuity_graph_against_model(graph, taxonomy, model)
    assert len(graph.chains) == 28
    assert graph.edge_count == 140
    assert graph.training_enabled_chain_count == 0
