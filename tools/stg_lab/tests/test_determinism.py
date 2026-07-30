from __future__ import annotations

from stg_lab.determinism import (
    compare_explicit_trajectory,
    merge_determinism_comparisons,
    run_explicit_trajectory,
)
from stg_lab.protocol import Action
from stg_lab.sim import SimulationConfig, STGEnvironment


class EmptyScenario:
    forecast_independent_of_player = True

    def __init__(self, name: str = "stage5_boss3", duration_frames: int = 10, x: float = 0.0):
        self.name = name
        self.scenario_key = f"{name}:lunatic"
        self.duration_frames = duration_frames
        self.x = x

    def reset(self, environment: STGEnvironment) -> None:
        environment.set_player_position(self.x, 0.0)

    def update(self, _environment: STGEnvironment) -> None:
        pass


def factory(name: str = "stage5_boss3", duration_frames: int = 10):
    def create(seed: int) -> STGEnvironment:
        return STGEnvironment(
            EmptyScenario(name, duration_frames),
            seed=seed,
            config=SimulationConfig(
                reaction_frames=0,
                action_hold_frames=1,
                semantic_width=8,
                semantic_height=8,
            ),
        )

    return create


def test_comparison_hashes_initial_frame_every_advance_and_actions() -> None:
    actions = (Action(move_x=1), Action(move_y=1), Action(slow=True), Action(move_x=-1))
    comparison = compare_explicit_trajectory(factory(), 3001, actions)

    assert comparison["matched"]
    assert comparison["initial_frame_included"]
    assert len(comparison["first_hashes"]) == len(actions) + 1
    assert comparison["first_hashes"] == comparison["second_hashes"]
    assert comparison["first_trajectory_hash"] == comparison["second_trajectory_hash"]
    assert comparison["actions"][0]["move_x"] == 1
    assert comparison["first"]["actions_consumed"] == len(actions)


def test_action_input_changes_the_trajectory_hash() -> None:
    stationary = run_explicit_trajectory(factory(), 3001, (Action(), Action()))
    moving = run_explicit_trajectory(factory(), 3001, (Action(move_x=1), Action()))

    assert stationary.frame_hashes[0] == moving.frame_hashes[0]
    assert stationary.frame_hashes[1] != moving.frame_hashes[1]
    assert stationary.trajectory_hash != moving.trajectory_hash


def test_termination_and_outcome_are_included() -> None:
    comparison = compare_explicit_trajectory(
        factory(duration_frames=2),
        3001,
        (Action(),) * 5,
    )

    assert comparison["matched"]
    assert comparison["first"]["frames"] == 2
    assert comparison["first"]["actions_consumed"] == 2
    assert comparison["first"]["terminated"]
    assert comparison["first"]["outcome"] == "clear"
    assert len(comparison["first_hashes"]) == 3


def test_a_nondeterministic_fresh_factory_is_detected() -> None:
    calls = 0

    def varying_factory(seed: int) -> STGEnvironment:
        nonlocal calls
        calls += 1
        return STGEnvironment(
            EmptyScenario(x=float(calls)),
            seed=seed,
            config=SimulationConfig(semantic_width=8, semantic_height=8),
        )

    comparison = compare_explicit_trajectory(varying_factory, 3001, (Action(),))
    assert not comparison["matched"]
    assert comparison["first_hashes"][0] != comparison["second_hashes"][0]


def test_factory_must_return_two_distinct_environments() -> None:
    environment = factory()(3001)

    def reused(_seed: int) -> STGEnvironment:
        return environment

    try:
        compare_explicit_trajectory(reused, 3001, (Action(),))
    except ValueError as error:
        assert "fresh environment" in str(error)
    else:  # pragma: no cover - assertion message is clearer than pytest.raises here
        raise AssertionError("reused environments must be rejected")


def test_max_frames_requires_a_complete_explicit_sequence() -> None:
    try:
        compare_explicit_trajectory(factory(), 3001, (Action(),), max_frames=2)
    except ValueError as error:
        assert "shorter than max_frames" in str(error)
    else:  # pragma: no cover
        raise AssertionError("short explicit action sequence must be rejected")


def test_two_scenario_comparisons_merge_into_json_ready_evidence() -> None:
    boss3 = compare_explicit_trajectory(
        factory("stage5_boss3", duration_frames=3), 3001, (Action(),) * 3,
    )
    boss4 = compare_explicit_trajectory(
        factory("stage5_boss4", duration_frames=3), 3001, (Action(),) * 3,
    )
    evidence = merge_determinism_comparisons((boss3, boss4))

    assert evidence["passed"]
    assert evidence["hash_scope"] == "per_frame"
    assert [item["scenario"] for item in evidence["comparisons"]] == [
        "stage5_boss3:lunatic",
        "stage5_boss4:lunatic",
    ]
