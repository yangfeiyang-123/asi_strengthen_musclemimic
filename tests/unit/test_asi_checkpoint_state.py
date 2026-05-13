from types import SimpleNamespace

import jax.numpy as jnp

from musclemimic.algorithms.common.asi import FrameASIState
from musclemimic.algorithms.common.checkpoint_manager import OrbaxCheckpointManager
from musclemimic.algorithms.ppo.checkpoint import create_agent_state_from_orbax
from musclemimic.algorithms.ppo.config import PPOAgentState


def test_checkpoint_extracts_asi_state_with_train_state():
    train_state = SimpleNamespace(
        params={"p": jnp.asarray([1.0])},
        opt_state={},
        step=jnp.asarray(3),
        run_stats={},
    )
    asi_state = FrameASIState(
        logits=jnp.asarray([[0.1, -0.1]], dtype=jnp.float32),
        baseline=jnp.asarray([[1.0, 2.0]], dtype=jnp.float32),
    )
    agent_state = SimpleNamespace(train_state=train_state, asi_state=asi_state)

    extracted = OrbaxCheckpointManager._extract_train_state(object(), agent_state)

    assert "asi_state" in extracted
    assert jnp.allclose(extracted["asi_state"]["logits"], asi_state.logits)
    assert jnp.allclose(extracted["asi_state"]["baseline"], asi_state.baseline)


def test_orbax_loader_restores_asi_state_when_present():
    orbax_data = {
        "params": {},
        "opt_state": {},
        "step": 0,
        "run_stats": {},
        "asi_state": {
            "logits": jnp.asarray([[0.2, -0.2]], dtype=jnp.float32),
            "baseline": jnp.asarray([[3.0, 4.0]], dtype=jnp.float32),
        },
    }

    agent_state = create_agent_state_from_orbax(orbax_data)

    assert hasattr(agent_state, "asi_state")
    assert jnp.allclose(agent_state.asi_state.logits, orbax_data["asi_state"]["logits"])
    assert jnp.allclose(agent_state.asi_state.baseline, orbax_data["asi_state"]["baseline"])


def test_ppo_agent_state_serializes_asi_state_only_when_present():
    train_state = SimpleNamespace(params={}, opt_state={}, step=0, run_stats={})
    asi_state = FrameASIState(
        logits=jnp.asarray([[0.3]], dtype=jnp.float32),
        baseline=jnp.asarray([[0.4]], dtype=jnp.float32),
    )

    serialized = PPOAgentState(train_state=train_state, asi_state=asi_state).serialize()

    assert "asi_state" in serialized
    assert jnp.allclose(serialized["asi_state"]["logits"], asi_state.logits)
    assert jnp.allclose(serialized["asi_state"]["baseline"], asi_state.baseline)
    assert "asi_state" not in PPOAgentState(train_state=train_state).serialize()
