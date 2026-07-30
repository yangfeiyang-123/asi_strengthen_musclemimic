"""Source-bound joint reporting for physiology, synergy and EMG evidence."""

from .build_continuity_ablation_report import (
    ABLATION_EVIDENCE_SCHEMA_VERSION,
    ABLATION_REPORT_SCHEMA_VERSION,
    build_continuity_ablation_report,
)
from .build_joint_report import (
    JOINT_REPORT_SCHEMA_VERSION,
    ROLLOUT_METRICS_EVIDENCE_SCHEMA_VERSION,
    build_joint_report,
    build_rollout_metrics_evidence,
    joint_report_fingerprint,
    validate_rollout_metrics_evidence,
)
from .calibrate_continuity_reward import (
    CALIBRATION_SCHEMA_VERSION,
    build_baseline_rollout_evidence,
    build_continuity_reward_calibration,
    calibration_fingerprint,
    validate_baseline_rollout_against_manifest,
    validate_continuity_reward_calibration,
)
from .collect_continuity_baseline import (
    build_environment_manifest,
    build_heldout_split_manifest,
    build_rollout_identity,
    validate_diagnostics_collection_contract,
)
from .promote_continuity_chains import (
    BATCH_A_CHAIN_IDS,
    CHAIN_REVIEW_SCHEMA_VERSION,
    chain_review_fingerprint,
    promote_batch_a_continuity_graph,
    validate_chain_review,
)

__all__ = [
    "ABLATION_EVIDENCE_SCHEMA_VERSION",
    "ABLATION_REPORT_SCHEMA_VERSION",
    "BATCH_A_CHAIN_IDS",
    "CALIBRATION_SCHEMA_VERSION",
    "CHAIN_REVIEW_SCHEMA_VERSION",
    "JOINT_REPORT_SCHEMA_VERSION",
    "ROLLOUT_METRICS_EVIDENCE_SCHEMA_VERSION",
    "build_baseline_rollout_evidence",
    "build_continuity_ablation_report",
    "build_continuity_reward_calibration",
    "build_environment_manifest",
    "build_heldout_split_manifest",
    "build_joint_report",
    "build_rollout_identity",
    "build_rollout_metrics_evidence",
    "calibration_fingerprint",
    "chain_review_fingerprint",
    "joint_report_fingerprint",
    "promote_batch_a_continuity_graph",
    "validate_baseline_rollout_against_manifest",
    "validate_chain_review",
    "validate_continuity_reward_calibration",
    "validate_diagnostics_collection_contract",
    "validate_rollout_metrics_evidence",
]
