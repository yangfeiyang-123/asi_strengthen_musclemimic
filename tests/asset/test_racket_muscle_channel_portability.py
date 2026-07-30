"""Asset-bound exact-scene versus portable muscle-channel ABI checks."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from musclemimic.environments.humanoids.myofullbody import MyoFullBody
from musclemimic.environments.humanoids.myofullbody_racket import MyoFullBodyRacket
from musclemimic.physiology.anatomical_groups import (
    load_anatomical_taxonomy,
    validate_taxonomy_against_model,
)
from musclemimic.physiology.continuity_groups import (
    load_fascicle_continuity_graph,
    validate_continuity_graph_against_model,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def taxonomy():
    return load_anatomical_taxonomy(ROOT / "configs/physiology/myofullbody_354_muscle_taxonomy_curated_v1.json")


@pytest.fixture(scope="module")
def graph(taxonomy):
    return load_fascicle_continuity_graph(
        ROOT / "configs/physiology/myofullbody_354_fascicle_continuity_v1.json",
        taxonomy=taxonomy,
    )


def test_bare_exact_and_racket_portable_binding(taxonomy, graph):
    bare = MyoFullBody(disable_fingers=True)._model
    racket = MyoFullBodyRacket(disable_fingers=True)._model
    validate_taxonomy_against_model(taxonomy, bare)
    with pytest.raises(ValueError, match="runtime MuJoCo model hash"):
        validate_taxonomy_against_model(taxonomy, racket)
    validate_taxonomy_against_model(
        taxonomy,
        racket,
        compatibility="portable_muscle_channel_abi",
    )
    validate_continuity_graph_against_model(graph, taxonomy, racket)


def test_portable_binding_rejects_ctrlrange_actadr_target_and_order_tamper(taxonomy):
    model = MyoFullBodyRacket(disable_fingers=True)._model
    first_id = int(taxonomy.ordered_actuators[0]["actuator_id"])

    original_ctrlrange = np.asarray(model.actuator_ctrlrange[first_id]).copy()
    model.actuator_ctrlrange[first_id] = [0.0, 0.9]
    with pytest.raises(ValueError, match=r"ctrlrange|core fingerprint"):
        validate_taxonomy_against_model(
            taxonomy,
            model,
            compatibility="portable_muscle_channel_abi",
        )
    model.actuator_ctrlrange[first_id] = original_ctrlrange

    original_actadr = int(model.actuator_actadr[first_id])
    model.actuator_actadr[first_id] = int(model.actuator_actadr[first_id + 1])
    with pytest.raises(ValueError, match=r"activation|unique"):
        validate_taxonomy_against_model(
            taxonomy,
            model,
            compatibility="portable_muscle_channel_abi",
        )
    model.actuator_actadr[first_id] = original_actadr

    original_target = np.asarray(model.actuator_trnid[first_id]).copy()
    model.actuator_trnid[first_id] = model.actuator_trnid[first_id + 1]
    with pytest.raises(ValueError, match=r"core fingerprint|transmission target"):
        validate_taxonomy_against_model(
            taxonomy,
            model,
            compatibility="portable_muscle_channel_abi",
        )
    model.actuator_trnid[first_id] = original_target

    rows = list(taxonomy.ordered_actuators)
    rows[0], rows[1] = rows[1], rows[0]
    reordered = dataclasses.replace(taxonomy, ordered_actuators=tuple(rows))
    with pytest.raises(ValueError, match=r"core fingerprint|actuator id"):
        validate_taxonomy_against_model(
            reordered,
            model,
            compatibility="portable_muscle_channel_abi",
        )
