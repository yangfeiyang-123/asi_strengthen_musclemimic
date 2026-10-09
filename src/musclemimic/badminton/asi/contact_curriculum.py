from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class _Stage:
    name: str
    start_update: int
    foot_h: float
    foot_v: float
    graph: float


@dataclass(frozen=True)
class ContactCurriculumState:
    stages: tuple[_Stage, ...]
    stage_name: str
    foot_contact_height_w: float
    foot_contact_velocity_w: float
    body_graph_w: float


_DEFAULT_STAGES = (_Stage("disabled", 0, 0.0, 0.0, 0.0),)


def create_contact_curriculum(config: dict[str, Any]) -> ContactCurriculumState:
    raw = config.get("stages", [])
    if not raw:
        stages = _DEFAULT_STAGES
    else:
        stages = tuple(
            _Stage(
                name=str(s.get("name", f"stage_{i}")),
                start_update=int(s["start_update"]),
                foot_h=float(s.get("foot_h", 0.0)),
                foot_v=float(s.get("foot_v", 0.0)),
                graph=float(s.get("graph", 0.0)),
            )
            for i, s in enumerate(raw)
        )
    unsupported = [stage.name for stage in stages if stage.graph != 0.0]
    if unsupported:
        raise ValueError(
            "body-graph Laplacian reward is not implemented; graph must be 0.0 "
            f"for every contact curriculum stage (non-zero: {unsupported})"
        )
    first = stages[0]
    return ContactCurriculumState(
        stages=stages,
        stage_name=first.name,
        foot_contact_height_w=first.foot_h,
        foot_contact_velocity_w=first.foot_v,
        body_graph_w=first.graph,
    )


def update_contact_curriculum(
    state: ContactCurriculumState,
    update_count: int,
) -> ContactCurriculumState:
    active = state.stages[0]
    for stage in state.stages:
        if update_count >= stage.start_update:
            active = stage
    return replace(
        state,
        stage_name=active.name,
        foot_contact_height_w=active.foot_h,
        foot_contact_velocity_w=active.foot_v,
        body_graph_w=active.graph,
    )
