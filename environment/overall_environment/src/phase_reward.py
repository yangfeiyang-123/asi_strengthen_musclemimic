from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from environment.overall_environment.src.contact_graph import ContactGraphReport, contact_reward_terms


@dataclass(frozen=True)
class PhaseRewardConfig:
    impact_phase: float
    phase_tolerance: float
    crossed_net_bonus: float = 0.25
    early_contact_penalty: float = -0.25


@dataclass(frozen=True)
class PhaseRewardTerms:
    pre_impact: float
    impact: float
    post_impact: float
    penalties: float

    def total(self) -> float:
        return float(self.pre_impact + self.impact + self.post_impact + self.penalties)


def compute_phase_gated_hit_reward(
    *,
    phase: float,
    contact_graph: ContactGraphReport,
    shuttle_state: Mapping[str, object],
    config: PhaseRewardConfig,
) -> PhaseRewardTerms:
    in_window = abs(float(phase) - float(config.impact_phase)) <= float(config.phase_tolerance)
    contact_terms = contact_reward_terms(contact_graph)

    impact = 0.0
    penalties = 0.0
    if in_window:
        impact = contact_terms["stringbed"]
    elif contact_graph.stringbed_shuttle_active:
        penalties += float(config.early_contact_penalty)

    post_impact = float(config.crossed_net_bonus) if bool(shuttle_state.get("crossed_net", False)) else 0.0
    return PhaseRewardTerms(
        pre_impact=contact_terms["hand_handle"],
        impact=float(impact),
        post_impact=post_impact,
        penalties=float(penalties),
    )
