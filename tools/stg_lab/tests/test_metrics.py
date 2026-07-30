from stg_lab.metrics import (
    AcceptanceReport,
    EpisodeMetrics,
    state_hash,
)


def episode(survived: bool, agreement: float = 1.0) -> EpisodeMetrics:
    return EpisodeMetrics("test", 1, survived, 60, 0.1, 1.0, action_agreement=agreement)


def test_state_hash_is_order_stable_and_float_quantized() -> None:
    first = {"player": {"x": 1.00000001}, "threats": [{"id": 2}, {"id": 1}]}
    second = {"threats": [{"id": 1}, {"id": 2}], "player": {"x": 1.0}}
    assert state_hash(first) == state_hash(second)


def test_state_hash_preserves_boolean_types() -> None:
    assert state_hash({"value": False}) != state_hash({"value": 0})
    assert state_hash({"value": True}) != state_hash({"value": 1})


def test_acceptance_report_is_machine_decidable() -> None:
    report = AcceptanceReport(
        planner=[episode(True) for _ in range(20)],
        visual=[episode(True, 0.9) for _ in range(20)],
        memory_first_risk=1.0,
        memory_second_risk=0.6,
        deterministic=True,
    )
    assert report.passed
    assert report.summary()["memory_risk_improvement"] == 0.4
