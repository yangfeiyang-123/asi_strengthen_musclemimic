from __future__ import annotations

from .models import ActionProtocol, ActionSpec


def _action(
    action_id: str,
    name: str,
    category: str,
    duration: float,
    trials: int,
    blocks: int = 1,
    rest: float = 30.0,
    instruction: str = "听到提示后完成动作并回到稳定站立",
    racket: bool = False,
    shuttle: bool = False,
    jump: bool = False,
    hit: bool = False,
    foot: bool = False,
) -> ActionSpec:
    return ActionSpec(
        action_id=action_id,
        display_name_zh=name,
        category=category,  # type: ignore[arg-type]
        duration_s=duration,
        baseline_s=1.0,
        cue_s=0.5,
        post_s=1.0,
        target_valid_trials=trials,
        blocks=blocks,
        rest_between_trials_s=rest,
        rest_between_blocks_s=180.0,
        instruction_zh=instruction,
        includes_racket=racket,
        includes_shuttle=shuttle,
        includes_jump=jump,
        requires_racket_contact=hit,
        requires_foot_contact=foot,
    )


BADMINTON_PRIMITIVE_PROTOCOL_V1 = ActionProtocol(
    protocol_id="badminton_primitive_protocol_v1",
    version=1,
    description="Primitive and complete badminton actions for EMG muscle-synergy experiments",
    actions=(
        _action("quiet_stance", "安静站立", "primitive", 4.0, 5, instruction="自然站立、保持安静，不主动收缩"),
        _action("split_step", "垫步启动", "primitive", 3.0, 5, foot=True),
        _action("forehand_lunge_and_return", "正手跨步并回位", "primitive", 4.0, 5, foot=True),
        _action("single_leg_countermovement_jump_land", "单腿反向跳与落地", "primitive", 4.0, 5, rest=60.0, jump=True, foot=True),
        _action("trunk_rotation", "躯干旋转", "primitive", 4.0, 5),
        _action("shadow_forehand_high_clear", "正手高远球挥拍影子动作", "primitive", 4.0, 5, racket=True),
        _action("shadow_forehand_lift", "正手挑球挥拍影子动作", "primitive", 4.0, 5, racket=True),
        _action("china_jump_shadow", "中国跳影子动作", "primitive", 5.0, 5, rest=75.0, racket=True, jump=True, foot=True),
        _action("forehand_lift_footwork_shadow", "正手挑球步伐影子动作", "primitive", 5.0, 5, rest=45.0, racket=True, foot=True),
        _action("forehand_high_clear", "正手高远球", "complete", 4.0, 10, blocks=2, rest=45.0, racket=True, shuttle=True, hit=True, foot=True),
        _action("china_jump_high_clear", "中国跳高远球", "complete", 5.0, 8, blocks=2, rest=75.0, racket=True, shuttle=True, jump=True, hit=True, foot=True),
        _action("forehand_lift_footwork", "正手挑球步伐", "complete", 5.0, 10, blocks=2, rest=45.0, racket=True, shuttle=True, hit=True, foot=True),
    ),
)


PROTOCOLS = {BADMINTON_PRIMITIVE_PROTOCOL_V1.protocol_id: BADMINTON_PRIMITIVE_PROTOCOL_V1}


def get_protocol(protocol_id: str) -> ActionProtocol:
    try:
        return PROTOCOLS[protocol_id]
    except KeyError as exc:
        raise KeyError(f"Unknown protocol {protocol_id!r}; choose from {sorted(PROTOCOLS)}") from exc
