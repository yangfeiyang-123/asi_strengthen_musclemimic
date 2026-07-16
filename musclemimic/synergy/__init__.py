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
from musclemimic.synergy.fit import (
    SynergyFitConfig,
    build_regional_composite_artifact,
    fit_synergy_dataset,
    fit_synergy_region,
    load_synergy_split,
)
from musclemimic.synergy.metrics import global_vaf, local_vaf
from musclemimic.synergy.nmf import NMFResult, fit_nmf, transform_nmf
from musclemimic.synergy.residual_fit import (
    StructuredResidualFitConfig,
    fit_structured_residual_basis,
    load_residual_mask_contract,
)
from musclemimic.synergy.schema import SignalTransform, SynergySignal

__all__ = [
    "SUPPORTED_STD_MODES",
    "NMFResult",
    "SignalTransform",
    "StructuredResidualFitConfig",
    "SynergyBasisArtifact",
    "SynergyFitConfig",
    "SynergySignal",
    "build_regional_composite_artifact",
    "calibrate_exploration_std",
    "ctrl_to_unit_excitation",
    "fit_nmf",
    "fit_structured_residual_basis",
    "fit_synergy_dataset",
    "fit_synergy_region",
    "global_vaf",
    "load_residual_mask_contract",
    "load_synergy_basis",
    "load_synergy_split",
    "local_vaf",
    "physical_exploration_rms",
    "save_synergy_basis",
    "transform_nmf",
]
