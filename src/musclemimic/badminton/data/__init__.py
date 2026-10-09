"""Data contracts for badminton reference inputs."""

from musclemimic.badminton.data.event_lookup import (
    EventReferenceLookup,
    select_transition_coordinates,
    write_event_reference_bank_manifest,
)
from musclemimic.badminton.data.event_schema import (
    EventAnnotation,
    EventPhaseArrays,
    EventTimeline,
    ForehandPhase,
    load_event_timeline,
)
from musclemimic.badminton.data.racket_reference import (
    RacketReference,
    load_racket_reference,
    racket_reference_metrics,
)
from musclemimic.badminton.data.reference_bundle import (
    ReferenceBundle,
    load_reference_bundle,
    reference_bundle_fingerprint,
    validate_reference_bundle_fingerprint,
)

__all__ = [
    "EventAnnotation",
    "EventPhaseArrays",
    "EventReferenceLookup",
    "EventTimeline",
    "ForehandPhase",
    "RacketReference",
    "ReferenceBundle",
    "load_event_timeline",
    "load_racket_reference",
    "load_reference_bundle",
    "racket_reference_metrics",
    "reference_bundle_fingerprint",
    "select_transition_coordinates",
    "validate_reference_bundle_fingerprint",
    "write_event_reference_bank_manifest",
]
