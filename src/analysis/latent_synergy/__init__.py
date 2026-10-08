"""CPU-friendly latent/synergy analysis tools.

Every public metric validates finite, non-empty arrays and raises ``ValueError``
instead of emitting a plausible-looking report from incomplete evidence.
"""

from analysis.latent_synergy.effective_dimension import effective_dimension_report
from analysis.latent_synergy.jacobian_alignment import jacobian_alignment_report
from analysis.latent_synergy.representation_similarity import representation_report

__all__ = (
    "effective_dimension_report",
    "jacobian_alignment_report",
    "representation_report",
)
