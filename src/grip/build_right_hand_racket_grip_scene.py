from __future__ import annotations

import argparse
import importlib
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import musclemimic_models


def _load_path_helpers():
    if __package__ in {None, ""}:
        sys.path.append(str(Path(__file__).resolve().parents[2]))
    paths = importlib.import_module("src.grip.paths")
    return paths.racket_xml_path, paths.scene_xml_path


racket_xml_path, scene_xml_path = _load_path_helpers()

HAND_GRIP_SITES: tuple[tuple[str, str, tuple[float, float, float]], ...] = (
    ("lunate_r", "rh_palm_grip_site", (0.0, 0.0, 0.0)),
    ("distal_thumb_r", "rh_thumb_pad_site", (0.0, 0.0, 0.0)),
    ("2distph_r", "rh_index_pad_site", (0.0, 0.0, 0.0)),
    ("3distph_r", "rh_middle_pad_site", (0.0, 0.0, 0.0)),
    ("4distph_r", "rh_ring_pad_site", (0.0, 0.0, 0.0)),
    ("5distph_r", "rh_pinky_pad_site", (0.0, 0.0, 0.0)),
)

RACKET_HANDLE_SITES: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("handle_axis_start_site", (0.0, 0.02, 0.0)),
    ("handle_axis_end_site", (0.0, 0.16, 0.0)),
    ("racket_face_normal_site", (0.0, 0.09, 0.05)),
)

SITE_SIZE = (0.006, 0.006, 0.006)
HAND_SITE_RGBA = (0.1, 0.7, 1.0, 1.0)
RACKET_SITE_RGBA = (1.0, 0.3, 0.1, 1.0)


def _add_site(body: mujoco.MjsBody, name: str, pos: tuple[float, float, float], rgba: tuple[float, float, float, float]) -> None:
    if any(site.name == name for site in body.sites):
        return
    body.add_site(name=name, pos=pos, size=SITE_SIZE, rgba=rgba)


def _add_hand_grip_sites(spec: mujoco.MjSpec) -> None:
    for body_name, site_name, pos in HAND_GRIP_SITES:
        body = spec.body(body_name)
        if body is None:
            raise ValueError(f"missing MyoFullBody right-hand body: {body_name}")
        _add_site(body, site_name, pos, HAND_SITE_RGBA)


def _add_racket_handle_sites(spec: mujoco.MjSpec) -> None:
    body = spec.body("racket")
    if body is None:
        raise ValueError("missing racket body: racket")
    for site_name, pos in RACKET_HANDLE_SITES:
        _add_site(body, site_name, pos, RACKET_SITE_RGBA)


def _postprocess_attached_xml(path: Path, asset_dir: Path) -> None:
    tree = ET.parse(path)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", str(asset_dir))
        compiler.set("texturedir", str(asset_dir))
    for elem in root.iter():
        value = elem.attrib.get("name")
        if value is not None and value.startswith("/"):
            elem.set("name", value[1:])
    tree.write(path, encoding="utf-8", xml_declaration=True)


def build_scene(output_xml: Path | str | None = None) -> Path:
    out_path = Path(output_xml) if output_xml is not None else scene_xml_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    myofullbody_path = Path(musclemimic_models.get_xml_path("myofullbody"))
    base_spec = mujoco.MjSpec.from_file(str(myofullbody_path))
    racket_spec = mujoco.MjSpec.from_file(str(racket_xml_path()))

    _add_hand_grip_sites(base_spec)
    _add_racket_handle_sites(racket_spec)

    racket_frame = base_spec.worldbody.add_frame(name="racket_mount_frame")
    base_spec.attach(racket_spec, frame=racket_frame)

    base_spec.compile()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "right_hand_racket_grip_scene.xml"
        base_spec.to_file(str(tmp_path))
        _postprocess_attached_xml(tmp_path, myofullbody_path.parent.parent)
        mujoco.MjModel.from_xml_path(str(tmp_path))
        out_path.write_text(tmp_path.read_text(encoding="utf-8"), encoding="utf-8")

    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the right-hand racket grip MuJoCo scene.")
    parser.add_argument("--out", type=Path, default=scene_xml_path(), help="Output XML path.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out = build_scene(output_xml=args.out)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
