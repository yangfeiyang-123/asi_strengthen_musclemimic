from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import mujoco

NameCandidates: TypeAlias = dict[str, tuple[str, ...]]

RIGHT_HAND_BODY_CANDIDATES: NameCandidates = {
    "palm": ("lunate_r", "hand_r", "right_hand", "wrist_r"),
    "thumb": ("distal_thumb_r", "thumbdist_r", "proximal_thumb_r", "right_thumb_distal"),
    "index": ("2distph_r", "right_index_distal", "index_distal_r"),
    "middle": ("3distph_r", "right_middle_distal", "middle_distal_r"),
    "ring": ("4distph_r", "right_ring_distal", "ring_distal_r"),
    "pinky": ("5distph_r", "right_pinky_distal", "pinky_distal_r"),
    "wrist": ("lunate_r", "hand_r", "right_hand", "wrist_r"),
    "thumb_proximal": ("proximal_thumb_r", "thumbprox_r", "right_thumb_proximal"),
    "thumb_distal": ("distal_thumb_r", "thumbdist_r", "right_thumb_distal"),
}

RIGHT_HAND_JOINT_NAMES: tuple[str, ...] = (
    "cmc_flexion_r",
    "cmc_abduction_r",
    "mp_flexion_r",
    "ip_flexion_r",
    "mcp2_flexion_r",
    "mcp2_abduction_r",
    "pm2_flexion_r",
    "md2_flexion_r",
    "mcp3_flexion_r",
    "mcp3_abduction_r",
    "pm3_flexion_r",
    "md3_flexion_r",
    "mcp4_flexion_r",
    "mcp4_abduction_r",
    "pm4_flexion_r",
    "md4_flexion_r",
    "mcp5_flexion_r",
    "mcp5_abduction_r",
    "pm5_flexion_r",
    "md5_flexion_r",
)

RIGHT_HAND_ACTUATOR_CANDIDATES: NameCandidates = {
    "FDS5": ("FDS5", "FDS5_r"),
    "FDS4": ("FDS4", "FDS4_r"),
    "FDS3": ("FDS3", "FDS3_r"),
    "FDS2": ("FDS2", "FDS2_r"),
    "FDP5": ("FDP5", "FDP5_r"),
    "FDP4": ("FDP4", "FDP4_r"),
    "FDP3": ("FDP3", "FDP3_r"),
    "FDP2": ("FDP2", "FDP2_r"),
    "EDC5": ("EDC5", "EDC5_r"),
    "EDC4": ("EDC4", "EDC4_r"),
    "EDC3": ("EDC3", "EDC3_r"),
    "EDC2": ("EDC2", "EDC2_r"),
    "EDM": ("EDM", "EDM_r"),
    "EIP": ("EIP", "EIP_r"),
    "EPL": ("EPL", "EPL_r"),
    "EPB": ("EPB", "EPB_r"),
    "FPL": ("FPL", "FPL_r"),
    "APL": ("APL", "APL_r"),
    "OP": ("OP", "OP_r"),
    "RI2": ("RI2", "RI2_r"),
    "LU_RB2": ("LU_RB2", "LU_RB2_r"),
    "UI_UB2": ("UI_UB2", "UI_UB2_r"),
    "RI3": ("RI3", "RI3_r"),
    "LU_RB3": ("LU_RB3", "LU_RB3_r"),
    "UI_UB3": ("UI_UB3", "UI_UB3_r"),
    "RI4": ("RI4", "RI4_r"),
    "LU_RB4": ("LU_RB4", "LU_RB4_r"),
    "UI_UB4": ("UI_UB4", "UI_UB4_r"),
    "RI5": ("RI5", "RI5_r"),
    "LU_RB5": ("LU_RB5", "LU_RB5_r"),
    "UI_UB5": ("UI_UB5", "UI_UB5_r"),
}

HAND_SITE_CANDIDATES: NameCandidates = {
    "palm": ("rh_palm_grip_site", "right_palm_site", "palm_site", "hand_r_palm"),
    "thumb": ("rh_thumb_pad_site", "right_thumb_pad_site", "thumb_tip_site", "thumb_distal_site"),
    "index": ("rh_index_pad_site", "right_index_pad_site", "index_tip_site", "index_distal_site"),
    "middle": ("rh_middle_pad_site", "right_middle_pad_site", "middle_tip_site", "middle_distal_site"),
    "ring": ("rh_ring_pad_site", "right_ring_pad_site", "ring_tip_site", "ring_distal_site"),
    "pinky": ("rh_pinky_pad_site", "right_pinky_pad_site", "pinky_tip_site", "pinky_distal_site"),
}

RACKET_BODY_CANDIDATES: tuple[str, ...] = ("racket", "racket_handle", "badminton_racket", "right_racket")
RACKET_FREEJOINT_CANDIDATES: tuple[str, ...] = ("racket_free", "racket_freejoint", "racket_root")
RACKET_SITE_NAMES: tuple[str, ...] = (
    "grip_pose_site",
    "butt_site",
    "stringbed_center_site",
    "head_tip_site",
    "handle_axis_start_site",
    "handle_axis_end_site",
    "racket_face_normal_site",
)
HANDLE_GEOM_CANDIDATES: tuple[str, ...] = (
    "handle_grip",
    "handle_bevel_00",
    "handle_bevel_01",
    "handle_bevel_02",
    "handle_bevel_03",
    "handle_bevel_04",
    "handle_bevel_05",
    "handle_bevel_06",
    "handle_bevel_07",
    "racket_handle",
    "racket_handle_geom",
    "handle",
    "grip",
)


@dataclass(frozen=True)
class HandRacketModelMap:
    hand_bodies: dict[str, str]
    hand_sites: dict[str, str]
    right_hand_joint_names: tuple[str, ...]
    right_hand_actuator_names: tuple[str, ...]
    racket_body: str | None
    racket_freejoint: str | None
    racket_sites: tuple[str, ...]
    handle_geoms: tuple[str, ...]
    missing: dict[str, tuple[str, ...]]

    @property
    def ok(self) -> bool:
        return not self.missing


def _name_exists(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> bool:
    return mujoco.mj_name2id(model, obj_type, name) >= 0


def _first_existing_name(
    model: mujoco.MjModel,
    obj_type: mujoco.mjtObj,
    candidates: tuple[str, ...],
) -> str | None:
    for name in candidates:
        if _name_exists(model, obj_type, name):
            return name
    return None


def _resolve_candidate_map(
    model: mujoco.MjModel,
    obj_type: mujoco.mjtObj,
    candidates_by_key: NameCandidates,
) -> tuple[dict[str, str], dict[str, tuple[str, ...]]]:
    found: dict[str, str] = {}
    missing: dict[str, tuple[str, ...]] = {}
    for key, candidates in candidates_by_key.items():
        name = _first_existing_name(model, obj_type, candidates)
        if name is None:
            missing[key] = candidates
        else:
            found[key] = name
    return found, missing


def _existing_names(model: mujoco.MjModel, obj_type: mujoco.mjtObj, names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(name for name in names if _name_exists(model, obj_type, name))


def _missing_names(model: mujoco.MjModel, obj_type: mujoco.mjtObj, names: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(name for name in names if not _name_exists(model, obj_type, name))


def _resolve_freejoint(model: mujoco.MjModel) -> str | None:
    for name in RACKET_FREEJOINT_CANDIDATES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            continue
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return name
    return None


def _prefix_missing(prefix: str, missing: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    return {f"{prefix}.{key}": names for key, names in missing.items()}


def load_model_map(
    model: mujoco.MjModel,
    require_racket: bool = True,
    require_grip_sites: bool = True,
) -> HandRacketModelMap:
    hand_bodies, missing_hand_bodies = _resolve_candidate_map(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        RIGHT_HAND_BODY_CANDIDATES,
    )
    right_hand_joint_names = _existing_names(model, mujoco.mjtObj.mjOBJ_JOINT, RIGHT_HAND_JOINT_NAMES)
    missing_right_hand_joint_names = _missing_names(model, mujoco.mjtObj.mjOBJ_JOINT, RIGHT_HAND_JOINT_NAMES)
    right_hand_actuators, missing_right_hand_actuators = _resolve_candidate_map(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        RIGHT_HAND_ACTUATOR_CANDIDATES,
    )

    hand_sites: dict[str, str] = {}
    missing_hand_sites: dict[str, tuple[str, ...]] = {}
    if require_grip_sites:
        hand_sites, missing_hand_sites = _resolve_candidate_map(
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            HAND_SITE_CANDIDATES,
        )

    racket_body = None
    racket_freejoint = None
    racket_sites: tuple[str, ...] = ()
    handle_geoms: tuple[str, ...] = ()

    missing: dict[str, tuple[str, ...]] = {}
    missing.update(_prefix_missing("hand_bodies", missing_hand_bodies))
    if missing_right_hand_joint_names:
        missing["right_hand_joint_names"] = missing_right_hand_joint_names
    missing.update(_prefix_missing("right_hand_actuators", missing_right_hand_actuators))
    missing.update(_prefix_missing("hand_sites", missing_hand_sites))

    if require_racket:
        racket_body = _first_existing_name(model, mujoco.mjtObj.mjOBJ_BODY, RACKET_BODY_CANDIDATES)
        racket_freejoint = _resolve_freejoint(model)
        racket_sites = _existing_names(model, mujoco.mjtObj.mjOBJ_SITE, RACKET_SITE_NAMES)
        handle_geoms = _existing_names(model, mujoco.mjtObj.mjOBJ_GEOM, HANDLE_GEOM_CANDIDATES)

        missing_racket_sites = _missing_names(model, mujoco.mjtObj.mjOBJ_SITE, RACKET_SITE_NAMES)
        if racket_body is None:
            missing["racket_body"] = RACKET_BODY_CANDIDATES
        if racket_freejoint is None:
            missing["racket_freejoint"] = RACKET_FREEJOINT_CANDIDATES
        if missing_racket_sites:
            missing["racket_sites"] = missing_racket_sites
        if not handle_geoms:
            missing["handle_geoms"] = HANDLE_GEOM_CANDIDATES

    return HandRacketModelMap(
        hand_bodies=hand_bodies,
        hand_sites=hand_sites,
        right_hand_joint_names=right_hand_joint_names,
        right_hand_actuator_names=tuple(right_hand_actuators.values()),
        racket_body=racket_body,
        racket_freejoint=racket_freejoint,
        racket_sites=racket_sites,
        handle_geoms=handle_geoms,
        missing=missing,
    )


def _format_names(names: tuple[str, ...]) -> str:
    return ", ".join(names) if names else "<none>"


def _print_model_map(model_map: HandRacketModelMap) -> None:
    print(f"hand_bodies: {_format_names(tuple(model_map.hand_bodies.values()))}")
    print(f"hand_sites: {_format_names(tuple(model_map.hand_sites.values()))}")
    print(f"right_hand_joint_names: {_format_names(model_map.right_hand_joint_names)}")
    print(f"right_hand_actuator_names: {_format_names(model_map.right_hand_actuator_names)}")
    print(f"racket_body: {model_map.racket_body or '<missing>'}")
    print(f"racket_freejoint: {model_map.racket_freejoint or '<missing>'}")
    print(f"racket_sites: {_format_names(model_map.racket_sites)}")
    print(f"handle_geoms: {_format_names(model_map.handle_geoms)}")
    if model_map.missing:
        print("missing:")
        for key, names in model_map.missing.items():
            print(f"  {key}: {_format_names(names)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit right-hand and racket names in a MuJoCo XML model.")
    parser.add_argument("--xml", type=Path, required=True, help="Path to the MuJoCo XML model to audit.")
    parser.add_argument(
        "--allow-missing-grip-sites",
        action="store_true",
        help="Do not fail when hand grip pad sites are absent.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    model_map = load_model_map(model, require_grip_sites=not args.allow_missing_grip_sites)
    print("PASS hand/racket model map audit" if model_map.ok else "FAIL hand/racket model map audit")
    _print_model_map(model_map)
    return 0 if model_map.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
