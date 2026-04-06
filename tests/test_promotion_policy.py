from __future__ import annotations

from datetime import datetime, timedelta

from bayesian_t1dm.promotion_policy import TrustLevel, UniversalReadinessInputs, compare_promotion, evaluate_universal_readiness


def test_evaluate_universal_readiness_promotes_to_default_trusted_when_all_checks_pass():
    now = datetime(2026, 4, 5, 10, 0, 0)
    decision = evaluate_universal_readiness(
        UniversalReadinessInputs(
            artifact_family="status",
            source_quality_status="good",
            validation_passed=True,
            reproducible=True,
            generated_at=now - timedelta(hours=1),
            now=now,
            freshness_window=timedelta(hours=24),
            publication_allowed=True,
            default_trusted_allowed=True,
        )
    )

    assert decision.trust_level is TrustLevel.DEFAULT_TRUSTED
    assert decision.publish_latest is True
    assert decision.default_trusted is True
    assert decision.reasons == ()


def test_evaluate_universal_readiness_blocks_on_bad_source_quality():
    decision = evaluate_universal_readiness(
        UniversalReadinessInputs(
            artifact_family="run",
            source_quality_status="degraded",
        )
    )

    assert decision.trust_level is TrustLevel.DIAGNOSTICS_ONLY
    assert decision.blocked is True
    assert "source_quality_not_good" in decision.reasons


def test_evaluate_universal_readiness_demotes_stale_artifacts_to_research_only():
    now = datetime(2026, 4, 5, 10, 0, 0)
    decision = evaluate_universal_readiness(
        UniversalReadinessInputs(
            artifact_family="latent_meal",
            source_quality_status="good",
            validation_passed=True,
            reproducible=True,
            generated_at=now - timedelta(days=3),
            now=now,
            freshness_window=timedelta(hours=24),
        )
    )

    assert decision.trust_level is TrustLevel.RESEARCH_ONLY
    assert decision.publish_latest is False
    assert decision.stale is True
    assert "stale_artifact" in decision.reasons


def test_evaluate_universal_readiness_blocks_when_validation_fails():
    decision = evaluate_universal_readiness(
        UniversalReadinessInputs(
            artifact_family="forecast",
            source_quality_status="good",
            validation_passed=False,
        )
    )

    assert decision.trust_level is TrustLevel.DIAGNOSTICS_ONLY
    assert "validation_not_passed" in decision.reasons


def test_compare_promotion_marks_demotion_when_current_level_drops():
    now = datetime(2026, 4, 5, 10, 0, 0)
    current = evaluate_universal_readiness(
        UniversalReadinessInputs(
            artifact_family="status",
            source_quality_status="good",
            validation_passed=True,
            reproducible=True,
            generated_at=now - timedelta(days=2),
            now=now,
            freshness_window=timedelta(hours=24),
        )
    )

    transition = compare_promotion(previous_level=TrustLevel.PUBLISHED_LATEST, current=current)

    assert transition.demoted is True
    assert transition.promoted is False
    assert transition.current_level is TrustLevel.RESEARCH_ONLY
