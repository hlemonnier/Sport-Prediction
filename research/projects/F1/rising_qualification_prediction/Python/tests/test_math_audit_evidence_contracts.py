from dataclasses import replace
import math
import pytest

from math_promotion_fixtures import protocol_fixture, paired_fixture, strategy_fixture
from packages.sports_core.evaluation.probability_metrics import brier_score, log_loss
from packages.sports_core.evaluation.calibration import expected_calibration_error
from packages.sports_core.evaluation.ranking_metrics import mean_absolute_rank_error
from packages.f1.orchestration.evaluation_protocol import validate_event_partitions
from packages.f1.orchestration.model_promotion import evaluate_model_promotion, live_strategy_promotion_config
from packages.f1.orchestration.non_live_validation import evaluate_qualifying_promotion
from packages.f1.orchestration.runtime import parse_train_seasons
from packages.f1.features.assembly import build_training_data


@pytest.mark.parametrize("metric", [brier_score, log_loss, expected_calibration_error])
@pytest.mark.parametrize("p,y", [([], []), ([.5], []), ([.1,.9], [0]), ([math.nan],[1]), ([math.inf],[0]), ([-.1],[0]), ([1.1],[1]), ([.5],[.3]), ([.5],[math.nan])])
def test_binary_metrics_reject_invalid_evidence(metric,p,y):
    with pytest.raises(ValueError): metric(p,y)


def test_binary_metrics_accept_generators_and_known_scores():
    assert brier_score((p for p in [.25,.75]), (y for y in [0,1])) == .0625
    assert log_loss([.25,.75],[0,1]) == pytest.approx(-math.log(.75))
    assert expected_calibration_error([.25,.75],[0,1]) == .25


@pytest.mark.parametrize("pred,actual", [([],[]), (["A"],["A","B"]), (["A","A"],["A","B"]), (["A"],["B"])])
def test_rank_metric_rejects_missing_population(pred,actual):
    with pytest.raises(ValueError): mean_absolute_rank_error(pred,actual)


@pytest.mark.parametrize("seasons", [[2027], [2025,2027], [0], [2025.5]])
def test_future_or_invalid_training_season_rejected_before_provider(seasons):
    class Provider:
        def list_rounds(self,*args):
            pytest.fail("invalid training input must not reach provider")
    with pytest.raises(ValueError):
        build_training_data(Provider(),"race",seasons,2026,4,False)


def test_cli_future_season_rejected():
    with pytest.raises(ValueError): parse_train_seasons("2025,2027",2026,"all")


def test_partition_unknown_keys_cannot_bypass_chronology():
    issues=validate_event_partitions(development=["past"], selection=["2025:01"],calibration=["2024:01"],audit=["future"])
    assert "development_event_key_invalid" in issues
    assert "event_partition_order_invalid:calibration" in issues


def test_partition_aliases_cannot_hide_overlap():
    issues=validate_event_partitions(development=["2022:R1"],selection=["2022-01"],calibration=["2023:01"],audit=["2024:01"])
    assert "event_partition_overlap:development:selection" in issues


def _qualifying(events, protocol):
    return evaluate_qualifying_promotion(events,baseline_kendall=.5,candidate_kendall=.6,pole_non_regression=True,top3_non_regression=True,top10_non_regression=True,tail_excluded_delta=-.2,evaluation_protocol=protocol,bootstrap_samples=1000)


def test_two_event_zero_width_bootstrap_is_diagnostic_only():
    events=paired_fixture(2.,1.)[:2]
    protocol=replace(protocol_fixture(),audit=tuple(e.event_key for e in events))
    decision=_qualifying(events,protocol)
    assert decision.diagnostics.ci95_delta == (-1.,-1.)
    assert not decision.promoted
    assert "insufficient_independent_audit_events" in decision.reasons


@pytest.mark.parametrize("change,reason", [
    ({"development_exposed_events": ("2025:01",)},"audit_events_exposed_during_development"),
    ({"development_exposure_reviewed":False},"development_exposure_not_reviewed"),
    ({"frozen_at_utc":"2025-08-01T00:00:00Z"},"candidate_not_frozen_before_audit"),
    ({"audit_prediction_times_utc":{}},"freeze_or_audit_timing_missing"),
    ({"point_in_time_provenance_verified":False},"point_in_time_provenance_unverified"),
    ({"evaluated_candidate_sha256":"b"*64},"evaluated_candidate_changed_after_freeze"),
    ({"minimum_audit_events":2},"minimum_audit_events_below_design_floor"),
])
def test_good_scores_cannot_bypass_evaluation_protocol(change,reason):
    decision=_qualifying(paired_fixture(2.,1.), replace(protocol_fixture(),**change))
    assert not decision.promoted and reason in decision.reasons


def test_missing_protocol_is_rejected_and_honest_positive_control_passes():
    assert not _qualifying(paired_fixture(2.,1.),None).promoted
    assert _qualifying(paired_fixture(2.,1.),protocol_fixture()).promoted


def _strategy(evidence):
    return evaluate_model_promotion(candidate_model_id="policy",baseline_model_id="baseline",
      candidate_metrics={"policy_value":12.,"illegal_action_rate":0.,"regret_vs_oracle":1.},
      baseline_metrics={"policy_value":10.,"illegal_action_rate":.05,"regret_vs_oracle":2.},
      config=live_strategy_promotion_config(),evaluation_protocol=protocol_fixture(),paired_events=paired_fixture(2.,1.),simulator_validation_passed=True,strategy_evidence=evidence)


def test_simulator_success_cannot_replace_ope_or_prospective_shadow():
    assert not _strategy(None).promotion_gate_passed
    valid=strategy_fixture("policy")
    assert _strategy(valid).promotion_gate_passed
    invalid={**valid,"ope":{**valid["ope"],"effective_sample_size":1}}
    assert "strategy_ope_effective_sample_size_insufficient" in _strategy(invalid).reasons
    invalid={**valid,"live_shadow":{**valid["live_shadow"],"prospective":False}}
    assert "strategy_live_shadow_not_prospective_held_out" in _strategy(invalid).reasons


@pytest.mark.parametrize("change", [{"candidate_sha256":None},{"audit_prediction_times_utc":None},{"development":None}])
def test_malformed_protocol_fails_closed(change):
    assert replace(protocol_fixture(),**change).issues(protocol_fixture().audit)


def test_known_development_exposed_2026_events_cannot_be_redeclared_pristine():
    protocol=protocol_fixture(year=2026)
    assert "audit_events_exposed_during_development" in protocol.issues(protocol.audit)


def test_strategy_historical_or_missing_shadow_times_rejected():
    valid=strategy_fixture("policy")
    invalid={**valid,"live_shadow":{**valid["live_shadow"],"event_keys":["2020:01","2020:02","2020:03"]}}
    assert "strategy_live_shadow_not_after_evaluation_partitions" in _strategy(invalid).reasons
    invalid={**valid,"live_shadow":{**valid["live_shadow"],"event_prediction_times_utc":{}}}
    assert "strategy_live_shadow_event_timing_missing" in _strategy(invalid).reasons
    invalid={**valid,"ope":{**valid["ope"],"event_keys":None}}
    assert "strategy_evidence_payload_invalid" in _strategy(invalid).reasons


@pytest.mark.parametrize("year,round_number", [(2026.5,4),(2026,3.5),(float("inf"),1),(2026,True)])
def test_invalid_target_coordinates_rejected_before_provider(year,round_number):
    class Provider:
        def list_rounds(self,*args): pytest.fail("invalid coordinate reached provider")
    with pytest.raises(ValueError): build_training_data(Provider(),"race",[2025],year,round_number,False)


def test_textual_false_cannot_satisfy_non_regression_assertions():
    decision=evaluate_qualifying_promotion(paired_fixture(2.,1.),baseline_kendall=.5,candidate_kendall=.6,
       pole_non_regression="false",top3_non_regression="false",top10_non_regression="false",tail_excluded_delta=-.2,
       evaluation_protocol=protocol_fixture(),bootstrap_samples=1000)
    assert not decision.promoted
    assert all("gate_failed:"+field in decision.reasons for field in ("pole_non_regression","top3_non_regression","top10_non_regression"))
