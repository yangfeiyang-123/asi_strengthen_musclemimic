from __future__ import annotations

import pytest
import numpy as np
import subprocess
import sys

from environment.overall_environment.src.frozen_body_policy import (
    ActorCheckpointSpec,
    FrozenBodyPolicyArtifactError,
    FrozenBodyPolicy,
    FrozenBodyPolicyLoadError,
    export_frozen_body_policy,
    load_frozen_body_policy_manifest,
    reconstruct_actor_checkpoint_spec,
    restore_flax_actor_mean_for_verification,
    validate_actor_checkpoint_shapes,
)


CHECKPOINT = "checkpoints/de63059b16c0/checkpoint_7812"


def test_frozen_body_policy_load_accepts_adapter_schema_before_actor_restore():
    with pytest.raises(FrozenBodyPolicyLoadError, match="checkpoint actor restore is not implemented"):
        FrozenBodyPolicy.load(CHECKPOINT, overall_obs_size=283)


def test_reconstruct_actor_checkpoint_spec_from_metadata():
    spec = reconstruct_actor_checkpoint_spec(CHECKPOINT)

    assert spec == ActorCheckpointSpec(
        obs_size=2418,
        action_size=354,
        actor_hidden_layers=(1024,) * 12 + (2048, 2048, 2048, 1024),
        critic_hidden_layers=(1024,) * 12 + (2048, 2048, 2048, 1024),
        activation="silu",
        init_std=3.0,
        learnable_std=True,
        use_layernorm=True,
        layernorm_eps=1e-5,
    )


def test_validate_actor_checkpoint_shapes_matches_metadata():
    report = validate_actor_checkpoint_shapes(CHECKPOINT)

    assert report.valid is True
    assert report.actor_output_kernel_shape == (1024, 354)
    assert report.log_std_shape == (354,)
    assert report.run_stats_mean_shape == (2418,)
    assert report.reason == "actor checkpoint shapes match reconstructed spec"


def test_export_frozen_body_policy_metadata_only_writes_artifact_contract(tmp_path):
    manifest = export_frozen_body_policy(CHECKPOINT, tmp_path, restore_tensors=False)
    loaded = load_frozen_body_policy_manifest(tmp_path)

    assert manifest == loaded
    assert manifest.schema_version == 1
    assert manifest.tensor_format == "npz"
    assert manifest.has_tensors is False
    assert manifest.actor_spec.obs_size == 2418
    assert manifest.actor_spec.action_size == 354
    assert manifest.shape_report.valid is True
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "body_obs_schema.json").is_file()
    assert (tmp_path / "action_manifest.json").is_file()
    assert not (tmp_path / "params.npz").exists()
    assert not (tmp_path / "run_stats.npz").exists()


def test_load_from_export_requires_tensor_files(tmp_path):
    export_frozen_body_policy(CHECKPOINT, tmp_path, restore_tensors=False)

    with pytest.raises(FrozenBodyPolicyArtifactError, match="does not contain exported tensors"):
        FrozenBodyPolicy.load_from_export(tmp_path)


def test_load_from_export_outputs_finite_action_for_synthetic_artifact(tmp_path):
    spec = ActorCheckpointSpec(
        obs_size=3,
        action_size=2,
        actor_hidden_layers=(4,),
        critic_hidden_layers=(4,),
        activation="tanh",
        init_std=1.0,
        learnable_std=True,
        use_layernorm=False,
        layernorm_eps=1e-5,
    )
    FrozenBodyPolicy.write_synthetic_export(tmp_path, spec)

    policy = FrozenBodyPolicy.load_from_export(tmp_path)
    action = policy.act(np.zeros(3, dtype=float))

    assert action.shape == (2,)
    assert np.isfinite(action).all()


def test_exported_numpy_actor_matches_flax_actor_mean():
    policy = FrozenBodyPolicy.load_from_export("outputs/frozen_body_policy/de63059b16c0_7812")
    obs = np.linspace(-0.25, 0.25, policy.actor_spec.obs_size, dtype=np.float32)

    numpy_action = policy.act(obs)
    flax_action = restore_flax_actor_mean_for_verification(policy, obs)

    np.testing.assert_allclose(numpy_action, flax_action, rtol=1e-5, atol=1e-5)


def test_export_frozen_body_policy_cli_metadata_only(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "BadmintonMimic/scripts/export_frozen_body_policy.py",
            "--checkpoint",
            CHECKPOINT,
            "--out",
            str(tmp_path),
            "--metadata-only",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "has_tensors" in result.stdout
    manifest = load_frozen_body_policy_manifest(tmp_path)
    assert manifest.has_tensors is False
    assert manifest.actor_spec.obs_size == 2418
