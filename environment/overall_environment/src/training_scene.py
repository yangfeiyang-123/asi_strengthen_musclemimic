from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from environment.overall_environment.src.paths import OVERALL_ROOT


DEFAULT_REQUIRED_KEYFRAMES = ["overall_ready"]
DEFAULT_REQUIRED_SITES = ["rh_palm_grip_site", "overall_shuttle_com"]
DEFAULT_REQUIRED_GEOMS = [
    "overall_handle_grip",
    "overall_stringbed_ground_contact_proxy",
]


@dataclass(frozen=True)
class TrainingSceneReport:
    xml_path: str
    keyframes: list[str]
    actuator_count: int
    required_sites: list[str]
    missing_sites: list[str]
    required_geoms: list[str]
    missing_geoms: list[str]


def default_training_scene_path() -> Path:
    return OVERALL_ROOT / "assets" / "overall_badminton_training_scene.xml"


def build_training_scene_report(
    xml_path: str | Path,
    *,
    required_sites: list[str] | None = None,
    required_geoms: list[str] | None = None,
) -> TrainingSceneReport:
    import mujoco

    path = Path(xml_path)
    model = mujoco.MjModel.from_xml_path(str(path))
    keyframes = _names_for_type(model, mujoco.mjtObj.mjOBJ_KEY, model.nkey)
    site_names = set(_names_for_type(model, mujoco.mjtObj.mjOBJ_SITE, model.nsite))
    geom_names = set(_names_for_type(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom))

    required_sites = list(required_sites or DEFAULT_REQUIRED_SITES)
    required_geoms = list(required_geoms or DEFAULT_REQUIRED_GEOMS)

    return TrainingSceneReport(
        xml_path=str(path),
        keyframes=keyframes,
        actuator_count=int(model.nu),
        required_sites=required_sites,
        missing_sites=[name for name in required_sites if name not in site_names],
        required_geoms=required_geoms,
        missing_geoms=[name for name in required_geoms if name not in geom_names],
    )


def validate_training_scene_report(report: TrainingSceneReport) -> None:
    if "overall_ready" not in report.keyframes:
        raise ValueError("training scene missing overall_ready keyframe")
    if report.actuator_count <= 0:
        raise ValueError("training scene must expose actuators")
    if report.missing_sites:
        raise ValueError(f"training scene missing sites: {report.missing_sites}")
    if report.missing_geoms:
        raise ValueError(f"training scene missing geoms: {report.missing_geoms}")


def _names_for_type(model, obj_type, count: int) -> list[str]:
    import mujoco

    return [
        name
        for index in range(count)
        if (name := mujoco.mj_id2name(model, obj_type, index)) is not None
    ]
