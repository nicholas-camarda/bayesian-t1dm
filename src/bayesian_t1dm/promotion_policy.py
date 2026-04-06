from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any


class TrustLevel(StrEnum):
    DIAGNOSTICS_ONLY = "diagnostics_only"
    RESEARCH_ONLY = "research_only"
    PUBLISHED_LATEST = "published_latest"
    DEFAULT_TRUSTED = "default_trusted"


@dataclass(frozen=True)
class UniversalReadinessInputs:
    artifact_family: str
    source_quality_status: str = "unknown"
    validation_passed: bool = True
    reproducible: bool = True
    generated_at: datetime | None = None
    now: datetime | None = None
    freshness_window: timedelta | None = None
    publication_allowed: bool = True
    default_trusted_allowed: bool = False


@dataclass(frozen=True)
class GateCheckResult:
    name: str
    passed: bool
    reason_code: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class ReadinessDecision:
    artifact_family: str
    trust_level: TrustLevel
    publish_latest: bool
    default_trusted: bool
    blocked: bool
    stale: bool
    reasons: tuple[str, ...]
    checks: tuple[GateCheckResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_family": self.artifact_family,
            "trust_level": self.trust_level.value,
            "publish_latest": self.publish_latest,
            "default_trusted": self.default_trusted,
            "blocked": self.blocked,
            "stale": self.stale,
            "reasons": list(self.reasons),
            "checks": [asdict(check) for check in self.checks],
        }


@dataclass(frozen=True)
class PromotionTransition:
    previous_level: TrustLevel | None
    current_level: TrustLevel
    promoted: bool
    demoted: bool
    changed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "previous_level": None if self.previous_level is None else self.previous_level.value,
            "current_level": self.current_level.value,
            "promoted": self.promoted,
            "demoted": self.demoted,
            "changed": self.changed,
            "reasons": list(self.reasons),
        }


_LEVEL_ORDER = {
    TrustLevel.DIAGNOSTICS_ONLY: 0,
    TrustLevel.RESEARCH_ONLY: 1,
    TrustLevel.PUBLISHED_LATEST: 2,
    TrustLevel.DEFAULT_TRUSTED: 3,
}


def _is_stale(*, generated_at: datetime | None, now: datetime | None, freshness_window: timedelta | None) -> bool:
    if generated_at is None or now is None or freshness_window is None:
        return False
    return now - generated_at > freshness_window


def evaluate_universal_readiness(inputs: UniversalReadinessInputs) -> ReadinessDecision:
    stale = _is_stale(generated_at=inputs.generated_at, now=inputs.now, freshness_window=inputs.freshness_window)
    checks = (
        GateCheckResult(
            name="source_quality",
            passed=str(inputs.source_quality_status) == "good",
            reason_code=None if str(inputs.source_quality_status) == "good" else "source_quality_not_good",
            detail=f"source_quality_status={inputs.source_quality_status}",
        ),
        GateCheckResult(
            name="validation",
            passed=bool(inputs.validation_passed),
            reason_code=None if inputs.validation_passed else "validation_not_passed",
            detail=f"validation_passed={inputs.validation_passed}",
        ),
        GateCheckResult(
            name="reproducibility",
            passed=bool(inputs.reproducible),
            reason_code=None if inputs.reproducible else "not_reproducible",
            detail=f"reproducible={inputs.reproducible}",
        ),
        GateCheckResult(
            name="freshness",
            passed=not stale,
            reason_code=None if not stale else "stale_artifact",
            detail=None if not stale else "artifact exceeded freshness window",
        ),
    )
    reasons = tuple(check.reason_code for check in checks if check.reason_code is not None)
    blocked = any(reason in {"source_quality_not_good", "validation_not_passed", "not_reproducible"} for reason in reasons)

    if blocked:
        trust_level = TrustLevel.DIAGNOSTICS_ONLY
    elif stale:
        trust_level = TrustLevel.RESEARCH_ONLY
    elif inputs.default_trusted_allowed:
        trust_level = TrustLevel.DEFAULT_TRUSTED
    elif inputs.publication_allowed:
        trust_level = TrustLevel.PUBLISHED_LATEST
    else:
        trust_level = TrustLevel.RESEARCH_ONLY

    return ReadinessDecision(
        artifact_family=inputs.artifact_family,
        trust_level=trust_level,
        publish_latest=trust_level in {TrustLevel.PUBLISHED_LATEST, TrustLevel.DEFAULT_TRUSTED},
        default_trusted=trust_level is TrustLevel.DEFAULT_TRUSTED,
        blocked=blocked,
        stale=stale,
        reasons=reasons,
        checks=checks,
    )


def compare_promotion(*, previous_level: TrustLevel | None, current: ReadinessDecision) -> PromotionTransition:
    current_order = _LEVEL_ORDER[current.trust_level]
    previous_order = -1 if previous_level is None else _LEVEL_ORDER[previous_level]
    promoted = current_order > previous_order
    demoted = previous_level is not None and current_order < previous_order
    return PromotionTransition(
        previous_level=previous_level,
        current_level=current.trust_level,
        promoted=promoted,
        demoted=demoted,
        changed=promoted or demoted,
        reasons=current.reasons,
    )


__all__ = [
    "GateCheckResult",
    "PromotionTransition",
    "ReadinessDecision",
    "TrustLevel",
    "UniversalReadinessInputs",
    "compare_promotion",
    "evaluate_universal_readiness",
]
