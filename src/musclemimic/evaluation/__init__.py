"""Evaluation contracts for physiology and learned muscle representations."""

from importlib import import_module

__all__ = [
    "EmgFilterConfig",
    "build_physiology_report",
    "evaluate_emg_validation",
    "impact_aligned_resample",
    "kinetic_chain_metrics",
    "match_synergy_bases",
    "muscle_timing_metrics",
    "preprocess_emg",
    "synergy_residual_metrics",
    "validate_emg_mapping",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    module = (
        "physiology"
        if name
        in {
            "build_physiology_report",
            "kinetic_chain_metrics",
            "muscle_timing_metrics",
            "synergy_residual_metrics",
        }
        else "emg_eval"
    )
    return getattr(import_module(f"musclemimic.evaluation.{module}"), name)
