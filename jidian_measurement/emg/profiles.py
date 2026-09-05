from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .models import ChannelProfile, ChannelSpec


ProfileStatus = Literal[
    "active",
    "legacy_actual",
    "deprecated_misdeclared",
    "legacy_compatibility",
]
RetrospectiveMVCPolicy = Literal["never", "explicit_only", "allowed"]


@dataclass(frozen=True)
class ProfileRegistration:
    """Registry metadata kept outside immutable ChannelProfile snapshots."""

    profile: ChannelProfile
    status: ProfileStatus
    collection_allowed: bool
    analysis_allowed: bool
    retrospective_mvc_allowed: RetrospectiveMVCPolicy = "never"
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile.profile_id,
            "version": self.profile.version,
            "status": self.status,
            "collection_allowed": self.collection_allowed,
            "analysis_allowed": self.analysis_allowed,
            "retrospective_mvc_allowed": self.retrospective_mvc_allowed,
            "reason": self.reason,
        }


class ProfilePermissionError(ValueError):
    """Raised when a registered profile is used for a prohibited purpose."""


def _ch(sensor_id: int, side: str, slug: str, zh: str, en: str, abbr: str) -> ChannelSpec:
    return ChannelSpec(
        sensor_id=sensor_id,
        side=side,  # type: ignore[arg-type]
        muscle_slug=slug,
        name_zh=zh,
        name_en=en,
        abbreviation=abbr,
        mvc_instruction_zh=f"定位{zh}肌腹，按实验员给定方向进行最大等长抗阻并稳定保持3到5秒",
    )


# This declaration is intentionally immutable.  It was written into PILOT01 but
# did not describe PILOT01's physical placement.  Its registry entry below
# prevents any new collection or analysis under this ID.
BADMINTON_SYNERGY_16_V1 = ChannelProfile(
    profile_id="badminton_synergy_16_v1",
    version=1,
    description="16-channel right-handed badminton muscle-synergy profile; immutable after formal use",
    intended_handedness="right",
    channels=(
        _ch(1, "right", "upper_trapezius", "右斜方肌上束", "Upper Trapezius", "UT"),
        _ch(2, "right", "anterior_deltoid", "右三角肌前束", "Anterior Deltoid", "AD"),
        _ch(3, "right", "posterior_deltoid", "右三角肌后束", "Posterior Deltoid", "PD"),
        _ch(4, "right", "pectoralis_major_clavicular", "右胸大肌锁骨部", "Pectoralis Major, clavicular", "PM"),
        _ch(5, "right", "latissimus_dorsi", "右背阔肌", "Latissimus Dorsi", "LD"),
        _ch(6, "right", "triceps_lateral", "右肱三头肌外侧头", "Triceps Brachii Lateral", "TRI"),
        _ch(7, "right", "pronator_teres", "右旋前圆肌", "Pronator Teres", "PT"),
        _ch(8, "right", "extensor_carpi_radialis", "右桡侧腕伸肌", "Extensor Carpi Radialis", "ECR"),
        _ch(9, "right", "external_oblique", "右腹外斜肌", "External Oblique", "EO"),
        _ch(10, "left", "external_oblique", "左腹外斜肌", "External Oblique", "EO"),
        _ch(11, "right", "gluteus_maximus", "右臀大肌", "Gluteus Maximus", "GMAX"),
        _ch(12, "left", "gluteus_maximus", "左臀大肌", "Gluteus Maximus", "GMAX"),
        _ch(13, "right", "vastus_lateralis", "右股外侧肌", "Vastus Lateralis", "VL"),
        _ch(14, "left", "vastus_lateralis", "左股外侧肌", "Vastus Lateralis", "VL"),
        _ch(15, "right", "gastrocnemius_medialis", "右腓肠肌内侧头", "Gastrocnemius Medialis", "GM"),
        _ch(16, "left", "gastrocnemius_medialis", "左腓肠肌内侧头", "Gastrocnemius Medialis", "GM"),
    ),
)


BADMINTON_SYNERGY_16_LEGACY_ACTUAL_V1 = ChannelProfile(
    profile_id="badminton_synergy_16_legacy_actual_v1",
    version=1,
    description="Verified physical sensor placement used for the historical P001/PILOT01 collection",
    intended_handedness="right",
    channels=(
        _ch(1, "right", "upper_trapezius", "右斜方肌上束", "Upper Trapezius", "UT"),
        _ch(2, "right", "anterior_deltoid", "右三角肌前束", "Anterior Deltoid", "AD"),
        _ch(3, "right", "posterior_deltoid", "右三角肌后束", "Posterior Deltoid", "PD"),
        _ch(4, "right", "pectoralis_major_clavicular", "右胸大肌锁骨部", "Pectoralis Major, clavicular", "PM"),
        _ch(5, "right", "latissimus_dorsi", "右背阔肌", "Latissimus Dorsi", "LD"),
        _ch(6, "right", "triceps_lateral", "右肱三头肌外侧头", "Triceps Brachii Lateral", "TRI"),
        _ch(7, "right", "external_oblique", "右腹外斜肌", "External Oblique", "EO"),
        _ch(8, "left", "external_oblique", "左腹外斜肌", "External Oblique", "EO"),
        _ch(9, "right", "gluteus_maximus", "右臀大肌", "Gluteus Maximus", "GMAX"),
        _ch(10, "left", "gluteus_maximus", "左臀大肌", "Gluteus Maximus", "GMAX"),
        _ch(11, "right", "vastus_lateralis", "右股外侧肌", "Vastus Lateralis", "VL"),
        _ch(12, "left", "vastus_lateralis", "左股外侧肌", "Vastus Lateralis", "VL"),
        _ch(13, "right", "biceps_femoris_long_head", "右股二头肌长头", "Biceps Femoris Long Head", "BF"),
        _ch(14, "left", "biceps_femoris_long_head", "左股二头肌长头", "Biceps Femoris Long Head", "BF"),
        _ch(15, "right", "gastrocnemius_medialis", "右腓肠肌内侧头", "Gastrocnemius Medialis", "GM"),
        _ch(16, "left", "gastrocnemius_medialis", "左腓肠肌内侧头", "Gastrocnemius Medialis", "GM"),
    ),
)


BADMINTON_SYNERGY_16_V2 = ChannelProfile(
    profile_id="badminton_synergy_16_v2",
    version=2,
    description="Active 16-channel badminton synergy profile with forearm channels and no gluteus maximus channels",
    intended_handedness="right",
    channels=(
        _ch(1, "right", "upper_trapezius", "右斜方肌上束", "Upper Trapezius", "UT"),
        _ch(2, "right", "anterior_deltoid", "右三角肌前束", "Anterior Deltoid", "AD"),
        _ch(3, "right", "posterior_deltoid", "右三角肌后束", "Posterior Deltoid", "PD"),
        _ch(4, "right", "pectoralis_major_clavicular", "右胸大肌锁骨部", "Pectoralis Major, clavicular", "PM"),
        _ch(5, "right", "latissimus_dorsi", "右背阔肌", "Latissimus Dorsi", "LD"),
        _ch(6, "right", "triceps_lateral", "右肱三头肌外侧头", "Triceps Brachii Lateral", "TRI"),
        _ch(7, "right", "pronator_teres", "右旋前圆肌", "Pronator Teres", "PT"),
        _ch(8, "right", "extensor_carpi_radialis", "右桡侧腕伸肌", "Extensor Carpi Radialis", "ECR"),
        _ch(9, "right", "external_oblique", "右腹外斜肌", "External Oblique", "EO"),
        _ch(10, "left", "external_oblique", "左腹外斜肌", "External Oblique", "EO"),
        _ch(11, "right", "vastus_lateralis", "右股外侧肌", "Vastus Lateralis", "VL"),
        _ch(12, "left", "vastus_lateralis", "左股外侧肌", "Vastus Lateralis", "VL"),
        _ch(13, "right", "biceps_femoris_long_head", "右股二头肌长头", "Biceps Femoris Long Head", "BF"),
        _ch(14, "left", "biceps_femoris_long_head", "左股二头肌长头", "Biceps Femoris Long Head", "BF"),
        _ch(15, "right", "gastrocnemius_medialis", "右腓肠肌内侧头", "Gastrocnemius Medialis", "GM"),
        _ch(16, "left", "gastrocnemius_medialis", "左腓肠肌内侧头", "Gastrocnemius Medialis", "GM"),
    ),
)


LEGACY_HIGH_CLEAR_6CH = ChannelProfile(
    profile_id="legacy_high_clear_6ch",
    version=1,
    description="Historical sensors 1-6; do not combine with the 16-channel profile without explicit conversion",
    intended_handedness="legacy_unknown",
    channels=(
        _ch(1, "right", "gastrocnemius", "右腓肠肌（旧配置）", "Gastrocnemius", "GAS"),
        _ch(2, "right", "gluteus_maximus", "右臀大肌", "Gluteus Maximus", "GMAX"),
        _ch(3, "right", "flexor_carpi_ulnaris", "右尺侧腕屈肌", "Flexor Carpi Ulnaris", "FCU"),
        _ch(4, "right", "external_oblique", "右腹外斜肌", "External Oblique", "EO"),
        _ch(5, "right", "triceps_lateral", "右肱三头肌外侧头", "Triceps Brachii Lateral", "TRI"),
        _ch(6, "right", "pectoralis_major", "右胸大肌", "Pectoralis Major", "PM"),
    ),
)


PROFILE_REGISTRY = {
    BADMINTON_SYNERGY_16_V1.profile_id: ProfileRegistration(
        profile=BADMINTON_SYNERGY_16_V1,
        status="deprecated_misdeclared",
        collection_allowed=False,
        analysis_allowed=False,
        retrospective_mvc_allowed="never",
        reason="The declared channel map does not match the physical placement used in PILOT01.",
    ),
    BADMINTON_SYNERGY_16_LEGACY_ACTUAL_V1.profile_id: ProfileRegistration(
        profile=BADMINTON_SYNERGY_16_LEGACY_ACTUAL_V1,
        status="legacy_actual",
        collection_allowed=False,
        analysis_allowed=True,
        retrospective_mvc_allowed="explicit_only",
    ),
    BADMINTON_SYNERGY_16_V2.profile_id: ProfileRegistration(
        profile=BADMINTON_SYNERGY_16_V2,
        status="active",
        collection_allowed=True,
        analysis_allowed=True,
        retrospective_mvc_allowed="allowed",
    ),
    LEGACY_HIGH_CLEAR_6CH.profile_id: ProfileRegistration(
        profile=LEGACY_HIGH_CLEAR_6CH,
        status="legacy_compatibility",
        collection_allowed=True,
        analysis_allowed=True,
        retrospective_mvc_allowed="allowed",
        reason="Retained for the explicitly versioned historical six-channel workflow.",
    ),
}

# Compatibility mapping for callers that only need immutable profile objects.
PROFILES = {profile_id: registration.profile for profile_id, registration in PROFILE_REGISTRY.items()}


def get_profile_registration(profile_id: str) -> ProfileRegistration:
    try:
        return PROFILE_REGISTRY[profile_id]
    except KeyError as exc:
        raise KeyError(f"Unknown channel profile {profile_id!r}; choose from {sorted(PROFILE_REGISTRY)}") from exc


def get_profile(profile_id: str) -> ChannelProfile:
    return get_profile_registration(profile_id).profile


def require_collection_profile(profile_id: str) -> ChannelProfile:
    registration = get_profile_registration(profile_id)
    if not registration.collection_allowed:
        raise ProfilePermissionError(
            f"Profile {profile_id!r} has status={registration.status!r} and is not allowed for collection. "
            f"{registration.reason or ''}".rstrip()
        )
    return registration.profile


def require_analysis_profile(profile_id: str) -> ChannelProfile:
    registration = get_profile_registration(profile_id)
    if not registration.analysis_allowed:
        raise ProfilePermissionError(
            f"Profile {profile_id!r} has status={registration.status!r} and is not allowed for analysis. "
            f"{registration.reason or ''}".rstrip()
        )
    return registration.profile


def require_mvc_profile(profile_id: str, *, explicit_retrospective: bool = False) -> ChannelProfile:
    registration = get_profile_registration(profile_id)
    if registration.collection_allowed:
        return registration.profile
    if registration.retrospective_mvc_allowed == "explicit_only" and explicit_retrospective:
        return registration.profile
    if registration.retrospective_mvc_allowed == "explicit_only":
        raise ProfilePermissionError(
            f"Profile {profile_id!r} is historical and requires explicit retrospective MVC confirmation. "
            "Pass --allow-retrospective-profile-mvc after confirming unchanged sensor placement semantics."
        )
    raise ProfilePermissionError(
        f"Profile {profile_id!r} has status={registration.status!r} and cannot be used for MVC collection. "
        f"{registration.reason or ''}".rstrip()
    )
