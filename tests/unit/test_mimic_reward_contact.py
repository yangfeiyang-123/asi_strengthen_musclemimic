import numpy as np
import pytest


def test_contact_reward_computation_standalone():
    """Test the contact reward math directly (standalone, no env)."""
    from BadmintonMimic.asi.contact_tracking_data import ContactTrackingData

    ctd = ContactTrackingData(
        stance_mask=np.array([[True, False, True, False]], dtype=np.bool_),
        foot_points=np.array([[[0, 0, 0.02], [0, 0, 0.5], [0, 0, 0.01], [0, 0, 0.5]]], dtype=np.float32),
        body_laplacian=None,
        foot_labels=["l_ankle", "l_toe", "r_ankle", "r_toe"],
        num_frames=1,
        reference_fps=60.0,
        control_dt=0.01,
        effective_ref_stride=0.6,
    )
    stance = ctd.stance_mask[0]
    ref_z = ctd.foot_points[0, :, 2]
    actual_z = np.array([0.03, 0.5, 0.02, 0.5], dtype=np.float32)
    height_err = np.mean(np.abs(actual_z[stance] - ref_z[stance]))
    assert height_err == pytest.approx(0.01, abs=1e-4)
    reward = np.exp(-80.0 * height_err)
    assert 0.0 < reward < 1.0


def test_mimic_reward_class_accepts_contact_kwargs():
    """MimicReward.__init__ should accept contact kwargs without error."""
    import inspect
    from musclemimic.core.reward.trajectory_based import MimicReward

    sig = inspect.signature(MimicReward.__init__)
    src = inspect.getsource(MimicReward.__init__)
    assert "foot_contact_height_w_exp" in src
    assert "foot_contact_velocity_w_exp" in src
    assert "body_graph_w_exp" in src


def test_attach_contact_tracking_method_exists():
    from musclemimic.core.reward.trajectory_based import MimicReward

    assert hasattr(MimicReward, "attach_contact_tracking")
    assert callable(getattr(MimicReward, "attach_contact_tracking"))


def test_all_ctd_attrs_read_in_call_are_initialized_in_init():
    """Regression for P1: every self._ctd_* read in __call__ must be set in __init__,
    so non-contact configs never hit AttributeError on the reward hot path."""
    import re
    import inspect
    from musclemimic.core.reward.trajectory_based import MimicReward

    init_src = inspect.getsource(MimicReward.__init__)
    call_src = inspect.getsource(MimicReward.__call__)

    read_attrs = set(re.findall(r"self\.(_ctd_[a-zA-Z0-9_]+)", call_src))
    assigned_attrs = set(re.findall(r"self\.(_ctd_[a-zA-Z0-9_]+)\s*=", init_src))

    missing = read_attrs - assigned_attrs
    assert not missing, f"_ctd_ attrs read in __call__ but not initialized in __init__: {missing}"
