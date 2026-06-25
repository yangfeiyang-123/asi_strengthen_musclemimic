"""Latent muscle skill distillation and LAB utilities.

ActionMask is always available (pure numpy).  The JAX-dependent symbols
(networks, LAB, losses) are imported lazily so the package can be loaded
in environments that have numpy but not JAX.
"""

from musclemimic.latent_muscle.action_mask import ActionMask


def __getattr__(name: str):
    _lazy = {
        "LABActionOutput": "musclemimic.latent_muscle.lab",
        "LABActionWrapper": "musclemimic.latent_muscle.lab",
        "latent_action_barrier": "musclemimic.latent_muscle.lab",
        "latent_distillation_loss": "musclemimic.latent_muscle.losses",
        "ConditionalPrior": "musclemimic.latent_muscle.networks",
        "LatentDecoder": "musclemimic.latent_muscle.networks",
        "PosteriorEncoder": "musclemimic.latent_muscle.networks",
    }
    module_path = _lazy.get(name)
    if module_path is not None:
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
