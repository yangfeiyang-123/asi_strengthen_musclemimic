"""Per-action facts for the PEASD pipeline, in one place.

Before this module every stage rediscovered the action it was training on:
``data_qc`` imported a hardcoded 22/5 clip split, the launcher pinned
``forehandClear_standard`` in module constants, and the latent sweep fell back
to a forehand config whenever a caller forgot ``--base-config``.  That last one
is the failure mode worth naming: a ChinaJump run silently trained on forehand
clear's latent config and still reported success.

So the contract here is fail-closed.  An asset that does not exist yet is
``None``, and :meth:`ActionSpec.require` raises naming the missing field.  No
field ever falls back to another action's asset.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE1_PEASD_ARMS = ("T0", "T1", "T2", "T3", "T4")


@dataclass(frozen=True)
class ActionSpec:
    """Everything the pipeline needs to know about one badminton action."""

    action_id: str
    slug: str

    # --- retargeted trajectory data ------------------------------------
    data_variant: str
    source_namespace: str
    cache_namespace: str
    train_motions: tuple[str, ...]
    val_motions: tuple[str, ...]
    release_manifest: str

    # --- surface EMG ---------------------------------------------------
    # Trial directories under datasets/emg/processed/.  Several exist per
    # action because a session may be split into shadow and footwork blocks.
    emg_trial_actions: tuple[str, ...]

    # --- Stage 1 synergy prep ------------------------------------------
    env_prefix: str
    stage1_config: str
    synergy_grouping: str

    # Applicability is distinct from asset readiness.  For example, Stage 3 is
    # meaningful for ForehandLift but its calibrated PEASD task spec does not
    # exist yet; ChinaJump has no hitting endpoint at all.  Pipeline planners
    # must branch on these facts before requiring a stage-specific asset.
    stage1r_applicable: bool
    racket_applicable: bool
    stage3_applicable: bool

    # The latent phase vocabulary has to be produced by the action's actual
    # collection path.  An empty vocabulary with a null field is an explicit
    # not-yet-ready state, not permission to borrow Forehand Clear's six phases.
    latent_phase_field: str | None
    latent_phases: tuple[tuple[int, str], ...]
    latent_require_all_phases: bool

    # --- assets that may not exist yet; None means fail closed ---------
    coverage_phase_schema: str | None = None
    synergy_preset: str | None = None
    stage1r_config: str | None = None
    stage1r005_config: str | None = None
    stage2_config: str | None = None
    stage2_extend_config: str | None = None
    student_bc_config: str | None = None
    student_ppo_config: str | None = None
    latent_lab_config: str | None = None
    latent_synergy_config: str | None = None
    # Optional, action-owned Stage-1 PEASD-Lite matched-ablation assets.  The
    # tuple is labelled rather than positional so T4 can never be confused
    # with an event-shuffle control from a different experiment.
    stage1_peasd_configs: tuple[tuple[str, str], ...] | None = None
    racket_event_bank_config: str | None = None
    racket_mass_v2_configs: tuple[str, ...] | None = None
    stage3_spec: str | None = None
    stage3_v2_spec: str | None = None
    stage3_direct_spec: str | None = None
    racket_attachment: str | None = None

    def require(self, field_name: str) -> str:
        """Return an optional asset, or explain precisely what is missing."""
        known = {entry.name for entry in fields(self)}
        if field_name not in known:
            raise AttributeError(f"{field_name!r} is not an ActionSpec field")
        value = getattr(self, field_name)
        if value is None:
            raise ValueError(
                f"action {self.action_id!r} has no {field_name!r} asset yet. "
                f"Declare it in ACTIONS[{self.slug!r}] once the asset exists; "
                "do not substitute another action's asset."
            )
        return str(value)

    @property
    def dataset_root(self) -> Path:
        return REPO_ROOT / "datasets" / self.action_id

    @property
    def cache_variant(self) -> str:
        """Trailing component of ``cache_namespace`` (the on-disk cache dir)."""
        return self.cache_namespace.rsplit("/", 1)[-1]

    @property
    def source_bucket(self) -> str:
        """Leading component of ``source_namespace`` -- ``temp`` or ``wham``."""
        return self.source_namespace.split("/", 1)[0]

    @property
    def source_variant(self) -> str:
        return self.source_namespace.rsplit("/", 1)[-1]

    def motion_path(self, clip: str) -> str:
        """``rel_dataset_path`` for one clip, as training configs spell it."""
        return f"{self.action_id}/{self.cache_namespace}/{clip}"

    @property
    def train_motion_paths(self) -> tuple[str, ...]:
        return tuple(self.motion_path(name) for name in self.train_motions)

    @property
    def val_motion_paths(self) -> tuple[str, ...]:
        return tuple(self.motion_path(name) for name in self.val_motions)

    @property
    def all_motions(self) -> tuple[str, ...]:
        return (*self.train_motions, *self.val_motions)

    @property
    def latent_expected_val_motion_count(self) -> int:
        """Number of independently held-out motions required by latent eval."""
        return len(self.val_motions)

    @property
    def latent_phase_ready(self) -> bool:
        """Whether collection has an audited discrete phase field/vocabulary."""
        return self.latent_phase_field is not None and bool(self.latent_phases)

    @property
    def stage1_peasd_ready(self) -> bool:
        """Whether all five action-owned PEASD-Lite arm configs exist."""

        return self.stage1_peasd_configs is not None

    def stage1_peasd_config(self, arm: str) -> str:
        """Return one labelled matched-arm config, failing closed if absent."""

        label = str(arm).strip().upper()
        if label not in STAGE1_PEASD_ARMS:
            raise ValueError(
                f"unsupported Stage-1 PEASD arm {arm!r}; expected one of "
                f"{list(STAGE1_PEASD_ARMS)}"
            )
        if self.stage1_peasd_configs is None:
            raise ValueError(
                f"action {self.action_id!r} has no Stage-1 PEASD-Lite config assets; "
                "do not borrow another action's configs"
            )
        return dict(self.stage1_peasd_configs)[label]

    def env_var(self, suffix: str) -> str:
        return f"{self.env_prefix}_{suffix}"

    def validate(self) -> None:
        """Reject a split that overlaps or names a clip twice."""
        overlap = set(self.train_motions) & set(self.val_motions)
        if overlap:
            raise ValueError(
                f"{self.action_id}: train/validation split overlaps on {sorted(overlap)}"
            )
        for label, clips in (("train", self.train_motions), ("validation", self.val_motions)):
            if len(set(clips)) != len(clips):
                raise ValueError(f"{self.action_id}: duplicate clip in {label} split")
        if not self.train_motions or not self.val_motions:
            raise ValueError(f"{self.action_id}: both splits must be non-empty")

        phase_ids = [phase_id for phase_id, _name in self.latent_phases]
        phase_names = [name for _phase_id, name in self.latent_phases]
        if len(set(phase_ids)) != len(phase_ids) or len(set(phase_names)) != len(phase_names):
            raise ValueError(f"{self.action_id}: latent phase IDs and names must be unique")
        if any(type(phase_id) is not int or phase_id < 0 for phase_id in phase_ids):
            raise ValueError(f"{self.action_id}: latent phase IDs must be non-negative integers")
        if any(not str(name).strip() for name in phase_names):
            raise ValueError(f"{self.action_id}: latent phase names must be non-empty")
        if self.latent_phase_field is None:
            if self.latent_phases or self.latent_require_all_phases:
                raise ValueError(
                    f"{self.action_id}: a missing latent phase field requires an empty vocabulary "
                    "and require_all_phases=false"
                )
        elif not self.latent_phases:
            raise ValueError(f"{self.action_id}: latent phase field requires a vocabulary")

        if self.stage1_peasd_configs is not None:
            labels = tuple(label for label, _path in self.stage1_peasd_configs)
            paths = tuple(path for _label, path in self.stage1_peasd_configs)
            if labels != STAGE1_PEASD_ARMS:
                raise ValueError(
                    f"{self.action_id}: Stage-1 PEASD configs must be labelled "
                    f"exactly {STAGE1_PEASD_ARMS}"
                )
            if len(set(paths)) != len(paths) or any(not str(path).strip() for path in paths):
                raise ValueError(
                    f"{self.action_id}: Stage-1 PEASD config paths must be non-empty and unique"
                )
            if any(self.slug not in path for path in paths):
                raise ValueError(
                    f"{self.action_id}: every Stage-1 PEASD config must be action-owned "
                    f"and include slug {self.slug!r} in its path"
                )

        if not self.stage1r_applicable and any((self.stage1r_config, self.stage1r005_config)):
            raise ValueError(f"{self.action_id}: Stage1R assets declared for an inapplicable action")
        if self.stage1r_applicable and not all((self.stage1r_config, self.stage1r005_config)):
            raise ValueError(f"{self.action_id}: applicable Stage1R requires both 0.03/0.05 configs")

        racket_assets = (
            self.stage2_config,
            self.stage2_extend_config,
            self.student_bc_config,
            self.student_ppo_config,
            self.racket_event_bank_config,
            self.racket_mass_v2_configs,
        )
        if not self.racket_applicable and any(value is not None for value in racket_assets):
            raise ValueError(f"{self.action_id}: racket assets declared for an inapplicable action")
        if self.racket_applicable and not all(
            (self.stage2_config, self.stage2_extend_config, self.student_bc_config, self.student_ppo_config)
        ):
            raise ValueError(f"{self.action_id}: applicable legacy racket stage is incomplete")
        if (self.racket_event_bank_config is None) != (self.racket_mass_v2_configs is None):
            raise ValueError(f"{self.action_id}: event-bank and racket-mass-v2 configs must be declared together")
        if self.racket_mass_v2_configs is not None:
            if len(self.racket_mass_v2_configs) != 4:
                raise ValueError(f"{self.action_id}: racket-mass-v2 must declare four load rungs")
            for config, scale in zip(self.racket_mass_v2_configs, ("025", "050", "075", "100"), strict=True):
                if f"mass_{scale}" not in config:
                    raise ValueError(f"{self.action_id}: racket-mass-v2 rung order is invalid")

        stage3_assets = (self.stage3_spec, self.stage3_v2_spec, self.stage3_direct_spec, self.racket_attachment)
        if not self.stage3_applicable and any(value is not None for value in stage3_assets):
            raise ValueError(f"{self.action_id}: Stage3 assets declared for an inapplicable action")
        if self.stage3_applicable and not self.racket_applicable:
            raise ValueError(f"{self.action_id}: hitting Stage3 requires racket applicability")


_STAGE1 = "config_specific_task/stage1_body"
_STAGE2 = "config_specific_task/stage2_racket"
_STAGE2_V2 = "config_specific_task/stage2_racket_v2"
_DISTILL = "fullbody/config_specific_task/distill"
_STAGE1_PEASD = f"{_STAGE1}/peasd_lite_v1"
_CLEAR_GROUPING = "experiments/synergy/forehand_clear_myofullbody_354_regions_v1.json"
_ANATOMICAL_GROUPING = "experiments/synergy/myofullbody_354_anatomy_derived_regions_v1.json"


def _stage1_peasd_configs(slug: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (
            arm,
            f"{_STAGE1_PEASD}/conf_fullbody_{slug}_peasd_{arm.lower()}",
        )
        for arm in STAGE1_PEASD_ARMS
    )

FOREHAND_CLEAR = ActionSpec(
    action_id="forehandClear_standard",
    slug="forehand_clear",
    data_variant="raw_smooth_v1",
    source_namespace="temp/raw_smooth_v1",
    cache_namespace="muscle_trajectory/raw_smooth_v1",
    train_motions=(
        "6月2日(1)-10", "6月2日(1)-1", "6月2日(1)-2", "6月2日(1)-4", "6月2日(1)-6",
        "6月2日(1)-7", "6月2日(1)-8", "6月2日(1)-9", "6月2日-2", "6月2日-3", "6月2日-4",
        "6月2日-6", "6月2日-7", "video1", "video2", "video3", "video4", "video5",
        "video6", "video7", "video8", "video9",
    ),
    val_motions=("6月2日(1)-3", "6月2日(1)-5", "6月2日-1", "6月2日-5", "video10"),
    release_manifest="datasets/forehandClear_standard/manifests/raw_smooth_v1/release_manifest.json",
    emg_trial_actions=("forehand_high_clear",),
    env_prefix="MUSCLEMIMIC_FOREHAND_UNIFIED",
    stage1_config=f"{_STAGE1}/conf_fullbody_forehand_clear_body_local",
    # Keep the sealed main-action artifact path stable.  The action-neutral
    # audit copy below carries the same index partition for new actions.
    synergy_grouping=_CLEAR_GROUPING,
    stage1r_applicable=True,
    racket_applicable=True,
    stage3_applicable=True,
    latent_phase_field="phase_id",
    latent_phases=(
        (0, "ready"),
        (1, "backswing"),
        (2, "acceleration"),
        (3, "impact"),
        (4, "followthrough"),
        (5, "recovery"),
    ),
    latent_require_all_phases=True,
    synergy_preset="presets/forehand_early_unified_action_v4",
    stage1r_config=f"{_STAGE1}/conf_fullbody_forehand_clear_body_finger_isolated",
    stage1r005_config=f"{_STAGE1}/conf_fullbody_forehand_clear_body_finger_isolated_005",
    stage2_config=f"{_STAGE2}/conf_fullbody_badminton_racket_local",
    stage2_extend_config=f"{_STAGE2}/conf_fullbody_badminton_racket_local_extend_160m",
    student_bc_config=f"{_DISTILL}/conf_fullbody_forehandclear_racket_student_phase_bc.yaml",
    student_ppo_config="config_specific_task/distill/conf_fullbody_forehandclear_racket_student_phase_ppo",
    latent_lab_config=f"{_DISTILL}/latent_forehandclear_lab.yaml",
    latent_synergy_config=f"{_DISTILL}/latent_forehandclear_synergy_v3.yaml",
    stage1_peasd_configs=_stage1_peasd_configs("forehand_clear"),
    racket_event_bank_config=f"{_STAGE2_V2}/conf_fullbody_forehand_clear_racket_event_bank",
    racket_mass_v2_configs=tuple(
        f"{_STAGE2_V2}/conf_fullbody_forehand_clear_racket_mass_{scale}"
        for scale in ("025", "050", "075", "100")
    ),
    stage3_spec="experiments/posttrain/incoming_shuttle_hit_v1.yaml",
    stage3_v2_spec="experiments/posttrain/incoming_shuttle_hit_impact_recovery_v2.yaml",
    stage3_direct_spec="experiments/posttrain/incoming_shuttle_hit_full354_v1.yaml",
    racket_attachment="configs/racket_attachment/forehand_clear_rigid_v4_custom.json",
)

CHINA_JUMP = ActionSpec(
    action_id="ChinaJump",
    slug="chinajump",
    data_variant="optimized_qc10",
    # The only action whose retarget source lives under wham/ rather than temp/.
    source_namespace="wham/optimized_wham",
    cache_namespace="muscle_trajectory/optimized",
    train_motions=(
        "forehandJump-1", "forehandJump-2", "forehandJump-4", "forehandJump-6",
        "forehandJump-7", "forehandJump-13", "forehandJump-16", "forehandJump-18",
    ),
    val_motions=("forehandJump-8", "forehandJump-17"),
    release_manifest="datasets/ChinaJump/qc/optimized_qc_20260712.md",
    emg_trial_actions=("china_jump_high_clear",),
    env_prefix="MUSCLEMIMIC_CHINAJUMP",
    stage1_config=f"{_STAGE1}/conf_fullbody_chinajump_optimized_qc10",
    synergy_grouping=_ANATOMICAL_GROUPING,
    stage1r_applicable=False,
    racket_applicable=False,
    stage3_applicable=False,
    # The coverage schema records intended jump semantics, but no current
    # distill collector emits those IDs.  Keep the latent contract unready.
    latent_phase_field=None,
    latent_phases=(),
    latent_require_all_phases=False,
    coverage_phase_schema=(
        "fullbody/config_specific_task/stage1_body/chinajump_coverage_phase_schema_v1.json"
    ),
    latent_lab_config=f"{_DISTILL}/latent_chinajump_lab.yaml",
    latent_synergy_config=f"{_DISTILL}/latent_chinajump_synergy_v3.yaml",
    stage1_peasd_configs=_stage1_peasd_configs("chinajump"),
)

FOREHAND_LIFT = ActionSpec(
    action_id="forehandLift",
    slug="forehand_lift",
    data_variant="optimized_root_smooth_v2",
    source_namespace="temp/optimized_root_smooth_v2",
    cache_namespace="muscle_trajectory/optimized_root_smooth_v2",
    train_motions=(
        "forehandLift-1", "forehandLift-3", "forehandLift-4", "forehandLift-5",
        "5月13日-1", "5月13日-2", "5月13日-4", "5月13日-5", "5月13日-6",
        "5月13日-8", "5月13日-9", "5月13日-10",
    ),
    val_motions=("forehandLift-2", "forehandLift-6", "5月13日-3", "5月13日-7"),
    release_manifest="datasets/forehandLift/manifests/optimized_root_smooth_v2/release_manifest.json",
    # Two capture blocks: the racket lift itself plus its shadow-drill twin.
    emg_trial_actions=("forehand_lift_footwork", "shadow_forehand_lift"),
    env_prefix="MUSCLEMIMIC_FOREHAND_LIFT",
    stage1_config=f"{_STAGE1}/conf_fullbody_forehand_lift_optimized_root_smooth_v2",
    synergy_grouping=_ANATOMICAL_GROUPING,
    stage1r_applicable=True,
    racket_applicable=True,
    stage3_applicable=True,
    # No calibrated lift event-reference bank currently produces phase_id.
    # In particular, Forehand Clear's impact phase must not be transplanted.
    latent_phase_field=None,
    latent_phases=(),
    latent_require_all_phases=False,
    stage1r_config=f"{_STAGE1}/conf_fullbody_forehand_lift_body_finger_isolated",
    stage1r005_config=f"{_STAGE1}/conf_fullbody_forehand_lift_body_finger_isolated_005",
    stage2_config=f"{_STAGE2}/conf_fullbody_forehand_lift_racket_local",
    stage2_extend_config=f"{_STAGE2}/conf_fullbody_forehand_lift_racket_local_extend_160m",
    student_bc_config=f"{_DISTILL}/conf_fullbody_forehandlift_racket_student_phase_bc.yaml",
    student_ppo_config="config_specific_task/distill/conf_fullbody_forehandlift_racket_student_phase_ppo",
    latent_lab_config=f"{_DISTILL}/latent_forehandlift_lab.yaml",
    latent_synergy_config=f"{_DISTILL}/latent_forehandlift_synergy_v3.yaml",
    stage1_peasd_configs=_stage1_peasd_configs("forehand_lift"),
    # Stage2-v2 remains blocked on lift-specific event/contact calibration.
    racket_event_bank_config=None,
    racket_mass_v2_configs=None,
    # Hitting is applicable, but no calibrated PEASD lift target/feed task spec
    # exists.  The legacy ForehandNetLift spec is a different experiment.
    stage3_spec=None,
    stage3_v2_spec=None,
    stage3_direct_spec=None,
)

ACTIONS: dict[str, ActionSpec] = {
    spec.slug: spec for spec in (FOREHAND_CLEAR, CHINA_JUMP, FOREHAND_LIFT)
}
DEFAULT_ACTION = FOREHAND_CLEAR.slug

_ALIASES: dict[str, str] = {}
for _spec in ACTIONS.values():
    _ALIASES[_spec.slug.lower()] = _spec.slug
    _ALIASES[_spec.action_id.lower()] = _spec.slug
    for _trial in _spec.emg_trial_actions:
        _ALIASES.setdefault(_trial.lower(), _spec.slug)


def resolve(action: str) -> ActionSpec:
    """Look up an action by slug, dataset id, or EMG trial name."""
    key = _ALIASES.get(str(action).strip().lower())
    if key is None:
        raise ValueError(
            f"unknown action {action!r}; expected one of {sorted(ACTIONS)} "
            f"(or a dataset id / EMG trial alias)"
        )
    return ACTIONS[key]


def action_choices() -> tuple[str, ...]:
    """Slugs, for ``argparse(choices=...)``."""
    return tuple(sorted(ACTIONS))


def emg_trial_action_choices() -> tuple[str, ...]:
    """Every EMG trial directory name the three actions draw on."""
    return tuple(sorted({trial for spec in ACTIONS.values() for trial in spec.emg_trial_actions}))


for _spec in ACTIONS.values():
    _spec.validate()
del _spec
