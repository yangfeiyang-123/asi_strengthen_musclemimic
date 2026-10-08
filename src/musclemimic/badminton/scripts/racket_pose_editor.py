#!/usr/bin/env python3
"""Interactive global grip editor for a MyoFullBody right hand and racket.

Trajectories are read-only previews.  One fixed hand-racket transform and one
set of right-hand finger targets are saved as a fingerprinted global grip
preset that can be used by every training and validation trajectory.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from environment.overall_environment.src.racket_attachment import (
    DEFAULT_RACKET_ATTACHMENT_CONTRACT_PATH,
    REPO_ROOT,
    RacketAttachmentContract,
    canonical_contract_fingerprint,
    load_racket_attachment_contract,
)
from musclemimic.badminton.racket_grip_preset import (
    DEFAULT_RACKET_GRIP_PRESET_PATH,
    RIGHT_HAND_GRIP_JOINT_NAMES,
    RacketGripPreset,
    load_racket_grip_preset,
    write_racket_grip_preset,
)


DEFAULT_OUTPUT_CONTRACT = (
    REPO_ROOT / "configs" / "racket_attachment" / "forehand_clear_rigid_v5_custom.json"
)
DEFAULT_OUTPUT_PRESET = (
    REPO_ROOT / "configs" / "racket_grip" / "forehand_clear_grip_v3_custom.json"
)
_RACKET_HANDLE_AXIS = np.array([0.0, 1.0, 0.0], dtype=float)
_RACKET_FACE_NORMAL = np.array([0.0, 0.0, 1.0], dtype=float)

_FINGER_GROUPS = (
    ("拇指", RIGHT_HAND_GRIP_JOINT_NAMES[0:4]),
    ("食指", RIGHT_HAND_GRIP_JOINT_NAMES[4:8]),
    ("中指", RIGHT_HAND_GRIP_JOINT_NAMES[8:12]),
    ("无名指", RIGHT_HAND_GRIP_JOINT_NAMES[12:16]),
    ("小指", RIGHT_HAND_GRIP_JOINT_NAMES[16:20]),
)


def normalize_wxyz(quaternion: Any) -> np.ndarray:
    """Return a finite, unit, sign-canonicalized wxyz quaternion."""

    value = np.asarray(quaternion, dtype=float)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ValueError("quaternion must contain four finite wxyz values")
    norm = float(np.linalg.norm(value))
    if norm <= 1.0e-12:
        raise ValueError("quaternion norm must be positive")
    value = value / norm
    if value[0] < 0.0:
        value = -value
    return value


def validate_position_m(position_m: Any) -> np.ndarray:
    """Return a finite three-vector in the attachment hand's local frame."""

    value = np.asarray(position_m, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("position_m must contain three finite XYZ values")
    return value.copy()


def translate_position_m(position_m: Any, axis: int, delta_m: float) -> np.ndarray:
    """Translate a hand-local position without involving racket orientation."""

    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    if not math.isfinite(float(delta_m)):
        raise ValueError("delta_m must be finite")
    value = validate_position_m(position_m)
    value[axis] += float(delta_m)
    return value


def _wxyz_to_rotation(quaternion: Any) -> Rotation:
    wxyz = normalize_wxyz(quaternion)
    return Rotation.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])


def _rotation_to_wxyz(rotation: Rotation) -> np.ndarray:
    xyzw = rotation.as_quat()
    return normalize_wxyz([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def euler_xyz_degrees_to_wxyz(euler_xyz_degrees: Any) -> np.ndarray:
    euler = np.asarray(euler_xyz_degrees, dtype=float)
    if euler.shape != (3,) or not np.all(np.isfinite(euler)):
        raise ValueError("Euler angles must contain three finite XYZ values")
    return _rotation_to_wxyz(Rotation.from_euler("xyz", euler, degrees=True))


def wxyz_to_euler_xyz_degrees(quaternion: Any) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return _wxyz_to_rotation(quaternion).as_euler("xyz", degrees=True)


def rotate_about_racket_local_axis(
    quaternion: Any,
    axis: int,
    degrees: float,
) -> np.ndarray:
    """Post-compose a rotation about a racket-local axis."""

    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    if not math.isfinite(float(degrees)):
        raise ValueError("degrees must be finite")
    rotvec = np.zeros(3, dtype=float)
    rotvec[axis] = math.radians(float(degrees))
    return _rotation_to_wxyz(
        _wxyz_to_rotation(quaternion) * Rotation.from_rotvec(rotvec)
    )


def _rounded_vector(values: Any, width: int) -> list[float]:
    array = np.asarray(values, dtype=float)
    if array.shape != (width,) or not np.all(np.isfinite(array)):
        raise ValueError(f"expected {width} finite values")
    rounded = []
    for value in array:
        clean = round(float(value), 9)
        rounded.append(0.0 if clean == -0.0 else clean)
    return rounded


def build_adjusted_contract_document(
    source_contract: RacketAttachmentContract,
    *,
    position_m: Any,
    quaternion_wxyz: Any,
    contract_id: str,
) -> dict[str, Any]:
    """Create a canonical contract document with an updated hand-racket pose."""

    if not isinstance(contract_id, str) or not contract_id.strip():
        raise ValueError("contract_id must be a non-empty string")
    document = source_contract.to_payload()
    document["contract_id"] = contract_id.strip()
    document["relative_pose"] = {
        "position_m": _rounded_vector(position_m, 3),
        "quaternion_wxyz": _rounded_vector(normalize_wxyz(quaternion_wxyz), 4),
    }
    document["fingerprint"] = canonical_contract_fingerprint(document)
    return document


def write_adjusted_contract(
    source_contract: RacketAttachmentContract,
    output_path: str | Path,
    *,
    position_m: Any,
    quaternion_wxyz: Any,
    contract_id: str,
) -> RacketAttachmentContract:
    """Atomically write and strictly reload an adjusted contract."""

    output = Path(output_path).expanduser().resolve()
    if output == source_contract.source_path.resolve():
        raise ValueError("refusing to overwrite the source racket attachment contract")
    document = build_adjusted_contract_document(
        source_contract,
        position_m=position_m,
        quaternion_wxyz=quaternion_wxyz,
        contract_id=contract_id,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return load_racket_attachment_contract(output)


def _joint_qpos_width(model: mujoco.MjModel, joint_id: int) -> int:
    joint_type = int(model.jnt_type[joint_id])
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4
    return 1


def map_qpos_frames_by_joint_name(
    frames: np.ndarray,
    source_model: mujoco.MjModel,
    target_model: mujoco.MjModel,
    target_default_qpos: np.ndarray,
) -> np.ndarray:
    """Map qpos frames between fingerless/fingers-on models by joint name."""

    values = np.asarray(frames, dtype=float)
    if values.ndim != 2 or values.shape[1] != source_model.nq:
        raise ValueError(
            f"trajectory qpos must have shape [T, {source_model.nq}], got {values.shape}"
        )
    default = np.asarray(target_default_qpos, dtype=float)
    if default.shape != (target_model.nq,):
        raise ValueError("target_default_qpos has the wrong shape")
    mapped = np.repeat(default[None, :], values.shape[0], axis=0)
    matched = 0
    for source_joint in range(source_model.njnt):
        name = mujoco.mj_id2name(
            source_model,
            mujoco.mjtObj.mjOBJ_JOINT,
            source_joint,
        )
        if not name:
            continue
        target_joint = mujoco.mj_name2id(
            target_model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )
        if target_joint < 0:
            continue
        source_width = _joint_qpos_width(source_model, source_joint)
        target_width = _joint_qpos_width(target_model, target_joint)
        if source_width != target_width:
            raise ValueError(f"joint {name!r} changes qpos width across preview models")
        source_address = int(source_model.jnt_qposadr[source_joint])
        target_address = int(target_model.jnt_qposadr[target_joint])
        mapped[:, target_address : target_address + target_width] = values[
            :, source_address : source_address + source_width
        ]
        matched += 1
    if matched == 0:
        raise ValueError("trajectory and target model have no common named joints")
    return mapped


@dataclass(frozen=True)
class LoadedTrajectory:
    qpos: np.ndarray
    name: str
    source_path: str
    sha256: str
    frequency_hz: float | None


def _read_trajectory_payload(
    source: str | Path | io.BytesIO,
    *,
    name: str,
    source_path: str,
    sha256: str,
) -> LoadedTrajectory:
    with np.load(source, allow_pickle=False) as payload:
        if "qpos" not in payload:
            raise ValueError(f"trajectory has no qpos array: {source_path}")
        qpos = np.asarray(payload["qpos"], dtype=float)
        frequency_hz = None
        if "frequency" in payload:
            frequency_hz = float(np.asarray(payload["frequency"]).reshape(-1)[0])
    if qpos.ndim != 2 or qpos.shape[0] == 0 or not np.all(np.isfinite(qpos)):
        raise ValueError(
            f"trajectory qpos must be a non-empty finite [T, nq] array: {source_path}"
        )
    if frequency_hz is not None and (
        not math.isfinite(frequency_hz) or frequency_hz <= 0.0
    ):
        raise ValueError(f"trajectory frequency must be positive: {source_path}")
    return LoadedTrajectory(
        qpos=qpos,
        name=name,
        source_path=source_path,
        sha256=sha256,
        frequency_hz=frequency_hz,
    )


def load_trajectory_path(path: str | Path) -> LoadedTrajectory:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"trajectory not found: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return _read_trajectory_payload(
        resolved,
        name=resolved.stem,
        source_path=str(resolved),
        sha256=f"sha256:{digest.hexdigest()}",
    )


def load_trajectory_bytes(content: bytes, *, name: str) -> LoadedTrajectory:
    if not content:
        raise ValueError("uploaded trajectory is empty")
    return _read_trajectory_payload(
        io.BytesIO(content),
        name=Path(name).stem or "uploaded_trajectory",
        source_path=f"browser-upload:{name}",
        sha256=f"sha256:{hashlib.sha256(content).hexdigest()}",
    )


def _prepare_qpos_frames(
    raw_frames: np.ndarray,
    *,
    env: Any,
    contract_path: Path,
    finger_targets: np.ndarray | None = None,
) -> np.ndarray:
    """Map a trajectory to the fingers-on preview and apply one global grip."""

    from musclemimic.environments.humanoids.myofullbody_racket import MyoFullBodyRacket

    model = env._model
    default_qpos = np.asarray(env._data.qpos, dtype=float).copy()
    values = np.asarray(raw_frames, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("trajectory qpos must be a non-empty finite [T, nq] array")
    if values.shape[1] == model.nq:
        qpos_frames = values.copy()
    else:
        source_env = MyoFullBodyRacket(
            disable_fingers=True,
            racket_attachment_contract=contract_path,
            no_skybox=True,
        )
        if values.shape[1] != source_env._model.nq:
            raise ValueError(
                f"trajectory nq={values.shape[1]} matches neither fingers-on preview "
                f"nq={model.nq} nor fingerless training nq={source_env._model.nq}"
            )
        qpos_frames = map_qpos_frames_by_joint_name(
            values,
            source_env._model,
            model,
            default_qpos,
        )
    targets = (
        np.asarray(env.grip_finger_targets, dtype=float)
        if finger_targets is None
        else np.asarray(finger_targets, dtype=float)
    )
    if targets.shape != env.grip_finger_qpos_addrs.shape:
        raise ValueError("finger_targets shape does not match the preview model")
    qpos_frames[:, env.grip_finger_qpos_addrs] = targets
    return qpos_frames


@dataclass
class PreviewModel:
    env: Any
    model: mujoco.MjModel
    data: mujoco.MjData
    qpos_frames: np.ndarray
    initial_frame: int
    default_qpos: np.ndarray
    contract_path: Path
    trajectory: LoadedTrajectory | None


def prepare_preview_model(args: argparse.Namespace) -> PreviewModel:
    from musclemimic.environments.humanoids.myofullbody_racket import MyoFullBodyRacket

    env = MyoFullBodyRacket(
        disable_fingers=False,
        racket_attachment_contract=args.contract,
        racket_grip_preset=args.preset,
        no_skybox=True,
    )
    model = env._model
    data = mujoco.MjData(model)
    default_qpos = np.asarray(env._data.qpos, dtype=float).copy()

    trajectory = None
    if args.trajectory is None:
        qpos_frames = default_qpos[None, :].copy()
        qpos_frames[:, env.grip_finger_qpos_addrs] = env.grip_finger_targets
    else:
        trajectory = load_trajectory_path(args.trajectory)
        qpos_frames = _prepare_qpos_frames(
            trajectory.qpos,
            env=env,
            contract_path=args.contract,
        )

    initial_frame = int(args.frame)
    if initial_frame < 0 or initial_frame >= len(qpos_frames):
        raise ValueError(
            f"--frame {initial_frame} is outside trajectory range [0, {len(qpos_frames) - 1}]"
        )
    data.qpos[:] = qpos_frames[initial_frame]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return PreviewModel(
        env=env,
        model=model,
        data=data,
        qpos_frames=qpos_frames,
        initial_frame=initial_frame,
        default_qpos=default_qpos,
        contract_path=args.contract,
        trajectory=trajectory,
    )


class RacketPoseEditor:
    """Viser scene and GUI for one trajectory-independent grip state."""

    def __init__(
        self,
        preview: PreviewModel,
        source_contract: RacketAttachmentContract,
        *,
        source_preset: RacketGripPreset | None,
        output_contract_path: Path,
        output_preset_path: Path,
        contract_id: str,
        preset_id: str,
        host: str,
        port: int,
    ) -> None:
        import trimesh
        import viser

        from musclemimic.environments.humanoids.myofullbody_racket import RACKET_BODY_NAME
        from musclemimic.viewer.viser_utils import build_body_meshes

        self.preview = preview
        self.model = preview.model
        self.data = preview.data
        self.source_contract = source_contract
        self.source_preset = source_preset
        self.output_contract_path = output_contract_path.expanduser().resolve()
        self.output_preset_path = output_preset_path.expanduser().resolve()
        self.contract_id = contract_id
        self.preset_id = preset_id
        self.frame_index = preview.initial_frame
        self.position_m = validate_position_m(source_contract.relative_position_m)
        self.quaternion_wxyz = normalize_wxyz(
            source_contract.relative_quaternion_wxyz
        )
        self.finger_names = tuple(preview.env.grip_finger_names)
        self.finger_addrs = np.asarray(
            preview.env.grip_finger_qpos_addrs,
            dtype=int,
        )
        self.finger_targets = np.asarray(
            preview.env.grip_finger_targets,
            dtype=float,
        ).copy()
        if tuple(self.finger_names) != RIGHT_HAND_GRIP_JOINT_NAMES:
            raise ValueError(
                "preview model does not expose the expected 20 right-hand grip joints"
            )
        self.initial_position_m = self.position_m.copy()
        self.initial_quaternion_wxyz = self.quaternion_wxyz.copy()
        self.initial_finger_targets = self.finger_targets.copy()
        self._syncing = False

        self.racket_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            RACKET_BODY_NAME,
        )
        self.hand_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            source_contract.parent_body,
        )
        if min(self.racket_body_id, self.hand_body_id) < 0:
            raise ValueError("preview model is missing the racket or attachment hand body")

        self.server = viser.ViserServer(
            host=host,
            port=port,
            label="MuscleMimic 全局握拍编辑器",
        )
        self.server.scene.set_up_direction("+z")
        self.server.scene.configure_environment_map(environment_intensity=0.7)
        self.server.scene.add_grid(
            "/ground",
            width=8.0,
            height=8.0,
            plane="xy",
            cell_size=0.25,
            section_size=1.0,
        )

        visual_meshes = build_body_meshes(self.model, include_collision=False)
        collision_meshes = build_body_meshes(self.model, include_collision=True)
        self.body_handles: dict[int, Any] = {}
        with self.server.atomic():
            for body_id, mesh in visual_meshes.items():
                if self._is_racket_descendant(body_id):
                    continue
                self.body_handles[body_id] = self.server.scene.add_mesh_trimesh(
                    f"/human/body_{body_id}",
                    mesh,
                )

        racket_parts = [
            meshes[self.racket_body_id]
            for meshes in (visual_meshes, collision_meshes)
            if self.racket_body_id in meshes
        ]
        if not racket_parts:
            raise ValueError("racket model produced no Viser-compatible geometry")
        racket_mesh = (
            racket_parts[0]
            if len(racket_parts) == 1
            else trimesh.util.concatenate(racket_parts)
        )

        self.hand_frame = self.server.scene.add_frame(
            "/editor/hand_frame",
            axes_length=0.08,
            axes_radius=0.002,
            origin_radius=0.006,
        )
        self.gizmo = self.server.scene.add_transform_controls(
            "/editor/hand_frame/racket_pose",
            scale=0.12,
            line_width=3.0,
            disable_axes=True,
            disable_sliders=True,
            disable_rotations=False,
            depth_test=False,
            position=self.position_m,
            wxyz=self.quaternion_wxyz,
        )
        self.server.scene.add_mesh_trimesh(
            "/editor/hand_frame/racket_pose/racket_mesh",
            racket_mesh,
        )
        self.server.scene.add_frame(
            "/editor/hand_frame/racket_pose/stringbed_axes",
            position=source_contract.stringbed_position_m,
            wxyz=source_contract.stringbed_quaternion_wxyz,
            axes_length=0.075,
            axes_radius=0.0015,
            origin_radius=0.004,
        )
        self.server.scene.add_label(
            "/editor/hand_frame/racket_pose/handle_label",
            "拍柄 +Y",
            position=(0.0, 0.20, 0.0),
        )
        self.server.scene.add_label(
            "/editor/hand_frame/racket_pose/face_label",
            "拍面法向 +Z",
            position=(0.0, 0.532, 0.10),
        )
        self._setup_gui(viser)
        self._set_frame(self.frame_index)

        @self.server.on_client_connect
        def _(client: Any) -> None:
            self._focus_client(client)

    def _is_racket_descendant(self, body_id: int) -> bool:
        current = int(body_id)
        while current > 0:
            if current == self.racket_body_id:
                return True
            current = int(self.model.body_parentid[current])
        return False

    def _setup_gui(self, viser: Any) -> None:
        self.server.gui.add_markdown(
            "### 全局握拍状态\n"
            "轨迹和帧只用于检查效果，**不会按帧保存，也不会改写 NPZ**。"
            "这里调整的球拍相对手掌位置、朝向和 20 个右手手指角度会作为一个预设，"
            "统一用于所有训练与验证轨迹。"
        )
        self.status = self.server.gui.add_html("")

        with self.server.gui.add_folder("1. 选择预览轨迹"):
            initial_path = (
                self.preview.trajectory.source_path
                if self.preview.trajectory is not None
                and not self.preview.trajectory.source_path.startswith("browser-upload:")
                else ""
            )
            self.trajectory_path_input = self.server.gui.add_text(
                "服务器 NPZ 路径",
                initial_value=initial_path,
            )
            load_path_button = self.server.gui.add_button("加载服务器轨迹")

            @load_path_button.on_click
            def _(event: Any) -> None:
                try:
                    trajectory = load_trajectory_path(self.trajectory_path_input.value)
                    self._replace_trajectory(trajectory)
                except Exception as exc:
                    self._notify(event, "加载失败", str(exc), color="red")
                    return
                self._notify(
                    event,
                    "轨迹已加载（仅预览）",
                    f"{trajectory.name}: {len(trajectory.qpos)} 帧",
                    color="green",
                )

            upload_button = self.server.gui.add_upload_button(
                "从本机选择并上传 NPZ",
                mime_type="application/octet-stream,.npz",
            )

            @upload_button.on_upload
            def _(event: Any) -> None:
                uploaded = event.target.value
                try:
                    trajectory = load_trajectory_bytes(
                        uploaded.content,
                        name=uploaded.name,
                    )
                    self._replace_trajectory(trajectory)
                except Exception as exc:
                    self._notify(event, "上传加载失败", str(exc), color="red")
                    return
                self._notify(
                    event,
                    "上传轨迹已加载（仅预览）",
                    f"{trajectory.name}: {len(trajectory.qpos)} 帧",
                    color="green",
                )

            self.trajectory_info = self.server.gui.add_markdown("")
            self._update_trajectory_info()

        with self.server.gui.add_folder("2. 调整全局球拍位置与朝向"):
            self.position_input = self.server.gui.add_vector3(
                "球拍坐标 XYZ（手局部，m）",
                initial_value=tuple(self.position_m),
            )

            @self.position_input.on_update
            def _(event: Any) -> None:
                if self._syncing:
                    return
                self._set_position(event.target.value, update_input=False)

            self.translation_step_mm = self.server.gui.add_slider(
                "坐标步进（mm）",
                min=0.5,
                max=20.0,
                step=0.5,
                initial_value=2.0,
            )
            translate = self.server.gui.add_button_group(
                "坐标精调（不改变朝向）",
                options=["X−", "X+", "Y−", "Y+", "Z−", "Z+"],
            )

            @translate.on_click
            def _(event: Any) -> None:
                mapping = {
                    "X−": (0, -1.0),
                    "X+": (0, 1.0),
                    "Y−": (1, -1.0),
                    "Y+": (1, 1.0),
                    "Z−": (2, -1.0),
                    "Z+": (2, 1.0),
                }
                axis, sign = mapping[str(event.target.value)]
                self._set_position(
                    translate_position_m(
                        self.position_m,
                        axis,
                        sign * float(self.translation_step_mm.value) / 1000.0,
                    )
                )

            self.server.gui.add_markdown(
                "上面的 XYZ 是 `thirdmc_r` 手局部坐标。修改坐标只平移球拍，"
                "不会重新计算或改变下面的欧拉角/四元数。"
            )
            initial_euler = wxyz_to_euler_xyz_degrees(self.quaternion_wxyz)
            self.euler_sliders = []
            with self.server.gui.add_folder("手局部固定轴 XYZ 欧拉角（度）"):
                for axis, value in zip("XYZ", initial_euler, strict=True):
                    slider = self.server.gui.add_slider(
                        axis,
                        min=-180.0,
                        max=180.0,
                        step=0.25,
                        initial_value=float(value),
                    )

                    @slider.on_update
                    def _(_event: Any) -> None:
                        if self._syncing:
                            return
                        euler = [float(item.value) for item in self.euler_sliders]
                        self._set_rotation(euler_xyz_degrees_to_wxyz(euler))

                    self.euler_sliders.append(slider)

            self.nudge_degrees = self.server.gui.add_slider(
                "局部轴步进（度）",
                min=0.25,
                max=15.0,
                step=0.25,
                initial_value=2.0,
            )
            nudge = self.server.gui.add_button_group(
                "精调",
                options=["X−", "X+", "拍柄Y−", "拍柄Y+", "Z−", "Z+"],
            )

            @nudge.on_click
            def _(event: Any) -> None:
                mapping = {
                    "X−": (0, -1.0),
                    "X+": (0, 1.0),
                    "拍柄Y−": (1, -1.0),
                    "拍柄Y+": (1, 1.0),
                    "Z−": (2, -1.0),
                    "Z+": (2, 1.0),
                }
                axis, sign = mapping[str(event.target.value)]
                self._set_rotation(
                    rotate_about_racket_local_axis(
                        self.quaternion_wxyz,
                        axis,
                        sign * float(self.nudge_degrees.value),
                    )
                )

            flip = self.server.gui.add_button("拍柄轴翻转 180°")

            @flip.on_click
            def _(_event: Any) -> None:
                self._set_rotation(
                    rotate_about_racket_local_axis(
                        self.quaternion_wxyz,
                        1,
                        180.0,
                    )
                )

        self.finger_sliders: list[Any] = []
        with self.server.gui.add_folder("3. 调整全局右手握拍"):
            self.server.gui.add_markdown(
                "角度单位是度；保存时转换为弧度。所有滑块都会立刻应用到当前预览轨迹的每一帧。"
            )
            for group_name, joint_names in _FINGER_GROUPS:
                with self.server.gui.add_folder(group_name):
                    for joint_name in joint_names:
                        index = self.finger_names.index(joint_name)
                        lower, upper = self._joint_range_degrees(joint_name)
                        slider = self.server.gui.add_slider(
                            joint_name.removesuffix("_r"),
                            min=lower,
                            max=upper,
                            step=0.5,
                            initial_value=float(math.degrees(self.finger_targets[index])),
                        )

                        @slider.on_update
                        def _(_event: Any, joint_index: int = index) -> None:
                            if self._syncing:
                                return
                            self._set_finger_target(
                                joint_index,
                                math.radians(float(_event.target.value)),
                            )

                        self.finger_sliders.append(slider)

        self.frame_folder = self.server.gui.add_folder("4. 切换预览帧")
        self.frame_slider = None
        with self.frame_folder:
            navigation = self.server.gui.add_button_group(
                "帧导航",
                options=["上一帧", "下一帧"],
            )

            @navigation.on_click
            def _(event: Any) -> None:
                step = -1 if str(event.target.value) == "上一帧" else 1
                self._set_frame(
                    int(np.clip(self.frame_index + step, 0, len(self.preview.qpos_frames) - 1))
                )

        self._install_frame_slider()

        with self.server.gui.add_folder("5. 保存全局握拍预设"):
            self.output_contract_input = self.server.gui.add_text(
                "球拍 attachment 输出",
                initial_value=str(self.output_contract_path),
            )
            self.output_preset_input = self.server.gui.add_text(
                "全局握拍 preset 输出",
                initial_value=str(self.output_preset_path),
            )
            reset = self.server.gui.add_button_group(
                "重置编辑",
                options=["重置坐标", "重置朝向", "重置手指", "全部重置"],
            )

            @reset.on_click
            def _(event: Any) -> None:
                action = str(event.target.value)
                if action in {"重置坐标", "全部重置"}:
                    self._set_position(self.initial_position_m)
                if action in {"重置朝向", "全部重置"}:
                    self._set_rotation(self.initial_quaternion_wxyz)
                if action in {"重置手指", "全部重置"}:
                    self._set_all_finger_targets(self.initial_finger_targets)

            focus = self.server.gui.add_button("聚焦右手与球拍", icon=viser.Icon.FOCUS_2)

            @focus.on_click
            def _(event: Any) -> None:
                if event.client is not None:
                    self._focus_client(event.client)

            save = self.server.gui.add_button(
                "保存并应用于所有轨迹",
                color="blue",
                icon=viser.Icon.DEVICE_FLOPPY,
            )

            @save.on_click
            def _(event: Any) -> None:
                try:
                    contract, preset = self._save_global_preset()
                except Exception as exc:
                    self._notify(event, "保存失败", str(exc), color="red")
                    return
                self._notify(
                    event,
                    "全局握拍预设已保存",
                    f"{self._display_path(preset.source_path)}\n{preset.fingerprint}",
                    color="green",
                )
                self._update_status(
                    saved_contract_fingerprint=contract.fingerprint,
                    saved_preset_fingerprint=preset.fingerprint,
                )

        @self.gizmo.on_update
        def _(event: Any) -> None:
            if self._syncing:
                return
            self._set_rotation(event.target.wxyz, update_gizmo=False)

        self._update_status()

    def _joint_range_degrees(self, joint_name: str) -> tuple[float, float]:
        joint_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_JOINT,
            joint_name,
        )
        if joint_id < 0:
            raise ValueError(f"finger joint missing from preview model: {joint_name}")
        if bool(self.model.jnt_limited[joint_id]):
            lower, upper = np.degrees(self.model.jnt_range[joint_id])
        else:
            lower, upper = -180.0, 180.0
        target = math.degrees(self.finger_targets[self.finger_names.index(joint_name)])
        return float(min(lower, target)), float(max(upper, target))

    def _display_path(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(REPO_ROOT.resolve()))
        except ValueError:
            return str(path.resolve())

    def _notify(
        self,
        event: Any,
        title: str,
        body: str,
        *,
        color: str,
    ) -> None:
        if event.client is not None:
            event.client.add_notification(
                title,
                body,
                color=color,
                auto_close_seconds=8.0 if color != "red" else None,
            )

    def _update_trajectory_info(self) -> None:
        if not hasattr(self, "trajectory_info"):
            return
        trajectory = self.preview.trajectory
        if trajectory is None:
            self.trajectory_info.content = (
                "当前：**静态默认姿态（1 帧）**。可选择轨迹检查同一个握拍状态。"
            )
            return
        frequency = (
            "未知"
            if trajectory.frequency_hz is None
            else f"{trajectory.frequency_hz:g} Hz"
        )
        self.trajectory_info.content = (
            f"当前仅预览：**{trajectory.name}** · {len(self.preview.qpos_frames)} 帧 · "
            f"{frequency} · `{trajectory.sha256[:19]}…`"
        )

    def _install_frame_slider(self) -> None:
        if self.frame_slider is not None:
            self.frame_slider.remove()
            self.frame_slider = None
        if len(self.preview.qpos_frames) <= 1:
            return
        with self.frame_folder:
            slider = self.server.gui.add_slider(
                "预览轨迹帧",
                min=0,
                max=len(self.preview.qpos_frames) - 1,
                step=1,
                initial_value=self.frame_index,
            )

            @slider.on_update
            def _(event: Any) -> None:
                if self._syncing:
                    return
                self._set_frame(int(event.target.value))

            self.frame_slider = slider

    def _replace_trajectory(self, trajectory: LoadedTrajectory) -> None:
        qpos_frames = _prepare_qpos_frames(
            trajectory.qpos,
            env=self.preview.env,
            contract_path=self.preview.contract_path,
            finger_targets=self.finger_targets,
        )
        self.preview.qpos_frames = qpos_frames
        self.preview.trajectory = trajectory
        self.preview.initial_frame = 0
        self.frame_index = 0
        self._install_frame_slider()
        self._set_frame(0)
        self._update_trajectory_info()

    def _apply_finger_targets_to_all_frames(self) -> None:
        self.preview.qpos_frames[:, self.finger_addrs] = self.finger_targets

    def _set_finger_target(self, index: int, angle_rad: float) -> None:
        if index < 0 or index >= len(self.finger_targets):
            raise IndexError(f"finger target index out of range: {index}")
        if not math.isfinite(angle_rad):
            raise ValueError("finger angle must be finite")
        self.finger_targets[index] = float(angle_rad)
        self._apply_finger_targets_to_all_frames()
        self._set_frame(self.frame_index)

    def _set_all_finger_targets(self, targets: Any) -> None:
        values = np.asarray(targets, dtype=float)
        if values.shape != self.finger_targets.shape or not np.all(np.isfinite(values)):
            raise ValueError("finger targets must contain 20 finite angles")
        self.finger_targets[:] = values
        self._apply_finger_targets_to_all_frames()
        self._syncing = True
        try:
            for slider, value in zip(self.finger_sliders, values, strict=True):
                slider.value = float(math.degrees(value))
        finally:
            self._syncing = False
        self._set_frame(self.frame_index)

    def _set_position(
        self,
        position_m: Any,
        *,
        update_input: bool = True,
    ) -> None:
        """Update only hand-local translation; preserve racket orientation exactly."""

        quaternion_before = self.quaternion_wxyz.copy()
        self.position_m = validate_position_m(position_m)
        self._syncing = True
        try:
            self.gizmo.position = self.position_m
            if update_input:
                self.position_input.value = tuple(self.position_m)
        finally:
            self._syncing = False
        if not np.array_equal(self.quaternion_wxyz, quaternion_before):
            raise RuntimeError("position update unexpectedly changed racket orientation")
        self._update_status()

    def _set_rotation(
        self,
        quaternion: Any,
        *,
        update_gizmo: bool = True,
    ) -> None:
        self.quaternion_wxyz = normalize_wxyz(quaternion)
        self._syncing = True
        try:
            if update_gizmo:
                self.gizmo.wxyz = self.quaternion_wxyz
            self.gizmo.position = self.position_m
            euler = wxyz_to_euler_xyz_degrees(self.quaternion_wxyz)
            for slider, value in zip(self.euler_sliders, euler, strict=True):
                slider.value = float(value)
        finally:
            self._syncing = False
        self._update_status()

    def _set_frame(self, frame_index: int) -> None:
        self.frame_index = int(frame_index)
        if self.frame_index < 0 or self.frame_index >= len(self.preview.qpos_frames):
            raise IndexError(f"frame index out of range: {self.frame_index}")
        self.data.qpos[:] = self.preview.qpos_frames[self.frame_index]
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        with self.server.atomic():
            for body_id, handle in self.body_handles.items():
                handle.position = np.asarray(self.data.xpos[body_id], dtype=float)
                handle.wxyz = normalize_wxyz(self.data.xquat[body_id])
            self.hand_frame.position = np.asarray(
                self.data.xpos[self.hand_body_id],
                dtype=float,
            )
            self.hand_frame.wxyz = normalize_wxyz(self.data.xquat[self.hand_body_id])
            if self.frame_slider is not None:
                self._syncing = True
                try:
                    self.frame_slider.value = self.frame_index
                finally:
                    self._syncing = False
        self.server.flush()
        self._update_status()

    def _save_global_preset(self) -> tuple[RacketAttachmentContract, RacketGripPreset]:
        output_contract = Path(self.output_contract_input.value).expanduser().resolve()
        output_preset = Path(self.output_preset_input.value).expanduser().resolve()
        if (
            self.source_preset is not None
            and output_preset == self.source_preset.source_path.resolve()
        ):
            raise ValueError("refusing to overwrite the source grip preset; choose a new output")
        contract = write_adjusted_contract(
            self.source_contract,
            output_contract,
            position_m=self.position_m,
            quaternion_wxyz=self.quaternion_wxyz,
            contract_id=self.contract_id,
        )
        angles = {
            name: float(value)
            for name, value in zip(
                self.finger_names,
                self.finger_targets,
                strict=True,
            )
        }
        preset = write_racket_grip_preset(
            output_preset,
            preset_id=self.preset_id,
            attachment_contract=contract,
            finger_joint_angles_rad=angles,
        )
        self.output_contract_path = output_contract
        self.output_preset_path = output_preset
        return contract, preset

    def _focus_client(self, client: Any) -> None:
        hand = np.asarray(self.data.xpos[self.hand_body_id], dtype=float)
        hand_rotation = _wxyz_to_rotation(self.data.xquat[self.hand_body_id])
        racket_rotation = _wxyz_to_rotation(self.quaternion_wxyz)
        head_local = self.position_m + racket_rotation.apply([0.0, 0.53, 0.0])
        head_world = hand + hand_rotation.apply(head_local)
        center = 0.55 * hand + 0.45 * head_world
        client.camera.look_at = center
        client.camera.position = center + np.array([0.85, -0.85, 0.45])
        client.camera.up_direction = np.array([0.0, 0.0, 1.0])

    def _update_status(
        self,
        *,
        saved_contract_fingerprint: str | None = None,
        saved_preset_fingerprint: str | None = None,
    ) -> None:
        if not hasattr(self, "status"):
            return
        rotation = _wxyz_to_rotation(self.quaternion_wxyz)
        handle_axis = rotation.apply(_RACKET_HANDLE_AXIS)
        face_normal = rotation.apply(_RACKET_FACE_NORMAL)
        quaternion = ", ".join(f"{value:+.6f}" for value in self.quaternion_wxyz)
        position = ", ".join(f"{value:+.6f}" for value in self.position_m)
        normal = ", ".join(f"{value:+.3f}" for value in face_normal)
        handle = ", ".join(f"{value:+.3f}" for value in handle_axis)
        saved = ""
        if saved_contract_fingerprint and saved_preset_fingerprint:
            saved = (
                f"<br><b>最近 attachment:</b> <code>{saved_contract_fingerprint}</code>"
                f"<br><b>最近 preset:</b> <code>{saved_preset_fingerprint}</code>"
            )
        self.status.content = (
            f"<b>预览帧:</b> {self.frame_index}/{len(self.preview.qpos_frames) - 1}"
            " · <b>作用域:</b> 所有轨迹"
            f"<br><b>position XYZ m（手局部）:</b> <code>[{position}]</code>"
            f"<br><b>quaternion wxyz:</b> <code>[{quaternion}]</code>"
            f"<br><b>拍柄 +Y（手局部）:</b> <code>[{handle}]</code>"
            f"<br><b>拍面 +Z（手局部）:</b> <code>[{normal}]</code>"
            f"<br><b>手指目标:</b> {len(self.finger_targets)} 个关节（全局）"
            "<br><b>attachment 输出:</b> "
            f"<code>{self._display_path(self.output_contract_path)}</code>"
            f"<br><b>preset 输出:</b> <code>{self._display_path(self.output_preset_path)}</code>"
            f"{saved}"
        )

    def run(self) -> None:
        try:
            while True:
                time.sleep(0.25)
        except KeyboardInterrupt:
            pass
        finally:
            self.server.stop()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        type=Path,
        default=DEFAULT_RACKET_GRIP_PRESET_PATH,
        help="Global grip preset to edit. Defaults to the promoted v2 custom preset.",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=None,
        help="Source attachment contract. Defaults to the promoted preset binding.",
    )
    parser.add_argument(
        "--output",
        dest="output_contract",
        type=Path,
        default=DEFAULT_OUTPUT_CONTRACT,
        help="New attachment contract written by Save.",
    )
    parser.add_argument(
        "--preset-output",
        type=Path,
        default=DEFAULT_OUTPUT_PRESET,
        help="New all-trajectory grip preset written by Save.",
    )
    parser.add_argument(
        "--contract-id",
        default=None,
        help="ID stored in the output attachment contract.",
    )
    parser.add_argument(
        "--preset-id",
        default=None,
        help="ID stored in the output global grip preset.",
    )
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=None,
        help="Optional read-only training/GMR NPZ containing qpos.",
    )
    parser.add_argument("--frame", type=int, default=0, help="Initial preview frame.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Build and validate the real preview model, then exit without Viser.",
    )
    return parser


def _run_check(preview: PreviewModel) -> None:
    from musclemimic.environments.humanoids.myofullbody_racket import RACKET_BODY_NAME
    from musclemimic.viewer.viser_utils import build_body_meshes

    racket_body = mujoco.mj_name2id(
        preview.model,
        mujoco.mjtObj.mjOBJ_BODY,
        RACKET_BODY_NAME,
    )
    visual = build_body_meshes(preview.model, include_collision=False)
    collision = build_body_meshes(preview.model, include_collision=True)
    if racket_body < 0 or (racket_body not in visual and racket_body not in collision):
        raise RuntimeError("racket preview mesh could not be built")
    if len(preview.env.grip_finger_qpos_addrs) != len(RIGHT_HAND_GRIP_JOINT_NAMES):
        raise RuntimeError("global grip editor requires all 20 right-hand finger joints")
    print(
        "[OK] global racket grip editor preview "
        f"nq={preview.model.nq} frames={len(preview.qpos_frames)} "
        f"finger_targets={len(preview.env.grip_finger_targets)} "
        f"visual_bodies={len(visual)} collision_bodies={len(collision)}"
    )


def main() -> int:
    args = _build_parser().parse_args()
    source_preset = None
    if args.preset is not None:
        args.preset = args.preset.expanduser().resolve()
        source_preset = load_racket_grip_preset(args.preset)
    if args.contract is None:
        args.contract = (
            DEFAULT_RACKET_ATTACHMENT_CONTRACT_PATH
            if source_preset is None
            else source_preset.attachment_contract_path
        )
    args.contract = args.contract.expanduser().resolve()
    source_contract = load_racket_attachment_contract(args.contract)
    if (
        source_preset is not None
        and source_preset.attachment_contract_fingerprint != source_contract.fingerprint
    ):
        raise SystemExit("--contract does not match the attachment bound by --preset")

    args.output_contract = args.output_contract.expanduser().resolve()
    args.preset_output = args.preset_output.expanduser().resolve()
    if args.output_contract == source_contract.source_path.resolve():
        raise SystemExit("--output must differ from the source contract")
    if (
        source_preset is not None
        and args.preset_output == source_preset.source_path.resolve()
    ):
        raise SystemExit("--preset-output must differ from the source preset")

    preview = prepare_preview_model(args)
    if args.check:
        _run_check(preview)
        return 0

    contract_id = args.contract_id or args.output_contract.stem
    preset_id = args.preset_id or args.preset_output.stem
    editor = RacketPoseEditor(
        preview,
        source_contract,
        source_preset=source_preset,
        output_contract_path=args.output_contract,
        output_preset_path=args.preset_output,
        contract_id=contract_id,
        preset_id=preset_id,
        host=args.host,
        port=args.port,
    )
    shown_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    print(f"[READY] Open http://{shown_host}:{args.port}")
    print("        Choose any trajectory for preview; one saved grip preset applies to all.")
    print("        Ctrl-C stops the editor.")
    editor.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
