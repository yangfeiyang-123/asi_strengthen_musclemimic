"""Muscle-synergy extraction, validation, and artifact utilities."""

from musclemimic.synergy.basis_artifact import (
    SynergyBasisArtifact,
    load_synergy_basis,
    save_synergy_basis,
)
from musclemimic.synergy.collect import ctrl_to_unit_excitation
from musclemimic.synergy.fit import (
    SynergyFitConfig,
    build_regional_composite_artifact,
    fit_synergy_dataset,
    fit_synergy_region,
    load_synergy_split,
)
from musclemimic.synergy.metrics import global_vaf, local_vaf
from musclemimic.synergy.nmf import NMFResult, fit_nmf, transform_nmf
from musclemimic.synergy.schema import SignalTransform, SynergySignal

__all__ = [
    "NMFResult",
    "SignalTransform",
    "SynergyBasisArtifact",
    "SynergyFitConfig",
    "SynergySignal",
    "build_regional_composite_artifact",
    "ctrl_to_unit_excitation",
    "fit_nmf",
    "fit_synergy_dataset",
    "fit_synergy_region",
    "global_vaf",
    "load_synergy_basis",
    "load_synergy_split",
    "local_vaf",
    "save_synergy_basis",
    "transform_nmf",
]
