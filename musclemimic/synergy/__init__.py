"""Muscle-synergy extraction, validation, and artifact utilities."""

from musclemimic.synergy.basis_artifact import (
    SynergyBasisArtifact,
    load_synergy_basis,
    save_synergy_basis,
)
from musclemimic.synergy.collect import ctrl_to_unit_excitation
from musclemimic.synergy.exploration_scaling import (
    SUPPORTED_STD_MODES,
    calibrate_exploration_std,
    physical_exploration_rms,
)
from musclemimic.synergy.frozen_decoder import (
    FrozenBodyDecoder,
    decode_frozen_body_action,
    load_frozen_body_decoder,
    save_frozen_body_decoder,
)
from musclemimic.synergy.graph_nmf import (
    GraphRegularizationBinding,
    graph_regularization_lineage_fingerprint,
    load_verified_graph_regularization,
    validate_graph_regularization_manifest,
)
from musclemimic.synergy.metrics import global_vaf, local_vaf
from musclemimic.synergy.multistage_contract import (
    EXACT_RUNTIME_COMPATIBILITY,
    FIXED_SYNERGY_MODE,
    FIXED_SYNERGY_RESIDUAL_MODE,
    FULL_354_MODE,
    PORTABLE_COMPATIBILITY,
    BodySynergyContractV2,
    canonical_action_mode,
    load_body_synergy_contract,
    load_compatible_body_synergy_contract,
    save_body_synergy_contract,
)
from musclemimic.synergy.nmf import NMFResult, fit_nmf, transform_nmf
from musclemimic.synergy.schema import SignalTransform, SynergySignal

_LAZY_EXPORTS = {
    "SynergyFitConfig": ("musclemimic.synergy.fit", "SynergyFitConfig"),
    "build_regional_composite_artifact": (
        "musclemimic.synergy.fit",
        "build_regional_composite_artifact",
    ),
    "fit_synergy_dataset": ("musclemimic.synergy.fit", "fit_synergy_dataset"),
    "fit_synergy_region": ("musclemimic.synergy.fit", "fit_synergy_region"),
    "load_synergy_split": ("musclemimic.synergy.fit", "load_synergy_split"),
    "StructuredResidualFitConfig": (
        "musclemimic.synergy.residual_fit",
        "StructuredResidualFitConfig",
    ),
    "fit_structured_residual_basis": (
        "musclemimic.synergy.residual_fit",
        "fit_structured_residual_basis",
    ),
    "load_residual_mask_contract": (
        "musclemimic.synergy.residual_fit",
        "load_residual_mask_contract",
    ),
}


def __getattr__(name: str):
    """Load action-interface-dependent fitting APIs without import cycles."""

    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    from importlib import import_module

    value = getattr(import_module(target[0]), target[1])
    globals()[name] = value
    return value


__all__ = [
    "EXACT_RUNTIME_COMPATIBILITY",
    "FIXED_SYNERGY_MODE",
    "FIXED_SYNERGY_RESIDUAL_MODE",
    "FULL_354_MODE",
    "PORTABLE_COMPATIBILITY",
    "SUPPORTED_STD_MODES",
    "BodySynergyContractV2",
    "FrozenBodyDecoder",
    "GraphRegularizationBinding",
    "NMFResult",
    "SignalTransform",
    "StructuredResidualFitConfig",
    "SynergyBasisArtifact",
    "SynergyFitConfig",
    "SynergySignal",
    "build_regional_composite_artifact",
    "calibrate_exploration_std",
    "canonical_action_mode",
    "ctrl_to_unit_excitation",
    "decode_frozen_body_action",
    "fit_nmf",
    "fit_structured_residual_basis",
    "fit_synergy_dataset",
    "fit_synergy_region",
    "global_vaf",
    "graph_regularization_lineage_fingerprint",
    "load_body_synergy_contract",
    "load_compatible_body_synergy_contract",
    "load_frozen_body_decoder",
    "load_residual_mask_contract",
    "load_synergy_basis",
    "load_synergy_split",
    "load_verified_graph_regularization",
    "local_vaf",
    "physical_exploration_rms",
    "save_body_synergy_contract",
    "save_frozen_body_decoder",
    "save_synergy_basis",
    "transform_nmf",
    "validate_graph_regularization_manifest",
]
