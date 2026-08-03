from types import SimpleNamespace

import numpy as np
import pytest

from stg_lab.memory import EpisodicMemory
from stg_lab.protocol import Action
from stg_lab.rollout import (
    RolloutConfig,
    benchmark_second_attempt,
    collect_dagger_demonstrations,
    collect_demonstrations,
    evaluate_planner,
    evaluate_policy,
    evaluate_policy_variants,
    imminent_safe_actions,
    load_demonstrations,
    planner_teacher_action,
    scenario_memory_vector,
    shield_action_toward,
    teacher_action_agreement,
    _candidate_player_paths,
    _safe_action_endpoints,
)
from stg_lab.scenarios import make_environment
from stg_lab.sim import Bounds, SimulationConfig, STGEnvironment
from stg_lab.vision import VisionConfig


class EmptyScenario:
    name = "empty"

    def __init__(self, duration_frames: int = 6) -> None:
        self.duration_frames = duration_frames

    def reset(self, _environment: STGEnvironment) -> None:
        pass

    def update(self, _environment: STGEnvironment) -> None:
        pass


class RightHandHazard(EmptyScenario):
    name = "memory_hazard"

    def reset(self, environment: STGEnvironment) -> None:
        environment.spawn_circle(
            8.0,
            0.0,
            1.0,
            remove_outside=False,
            tag="test_hazard",
        )


class FixedPlanner:
    def plan(self, environment, *, observation=None):
        return SimpleNamespace(
            first_action=Action(),
            start_risk=abs(environment.player.x) / 4.0,
        )


VISION = VisionConfig(
    global_width=8,
    global_height=8,
    local_width=8,
    local_height=8,
    history=2,
    observation_delay=1,
)


def test_zero_width_scenario_memory_leaves_phase_to_recurrent_state() -> None:
    assert scenario_memory_vector("stage5_boss4:lunatic", 0).shape == (0,)
    assert np.array_equal(
        scenario_memory_vector("unseen_attack", 1),
        np.zeros(1, dtype=np.float32),
    )
    with pytest.raises(ValueError, match="cannot be negative"):
        scenario_memory_vector("anything", -1)


def test_scenario_vocabulary_uses_one_hot_identity_and_unknown_token() -> None:
    vocabulary = ("<unknown>", "attack:okuu:Lunatic#3")
    np.testing.assert_array_equal(
        scenario_memory_vector(
            "attack:okuu:Lunatic#3", 2, vocabulary,
        ),
        (0.0, 1.0),
    )
    np.testing.assert_array_equal(
        scenario_memory_vector("attack:new:Lunatic#1", 2, vocabulary),
        (1.0, 0.0),
    )
    with pytest.raises(ValueError, match="width"):
        scenario_memory_vector("anything", 1, vocabulary)


def test_policy_behavior_can_defer_runtime_commit_for_native_overrides(
    monkeypatch,
) -> None:
    from stg_lab import rollout
    from stg_lab.policy import ProficiencyRuntime

    class RecordingRuntime(ProficiencyRuntime):
        def __init__(self) -> None:
            super().__init__("expert", seed=7)
            self.commits = []

        def commit(self, action: Action, *, decision_interval: int) -> None:
            self.commits.append((action, decision_interval))
            super().commit(action, decision_interval=decision_interval)

    logits = np.full(18, -1.0, dtype=np.float32)
    logits[Action(move_x=1, slow=True).discrete] = 1.0
    monkeypatch.setattr(
        rollout,
        "_policy_logits",
        lambda *_args, **_options: (logits, None),
    )
    visible = rollout.VisionObservation(
        global_frames=np.zeros((1, 6, 8, 8), dtype=np.float32),
        local_frames=np.zeros((1, 6, 8, 8), dtype=np.float32),
        source_frame=0,
    )
    runtime = RecordingRuntime()
    options = {
        "device": "cpu",
        "memory": np.zeros(0, dtype=np.float32),
        "hidden": None,
        "inference_mode": "stream",
        "config": RolloutConfig(decision_interval=3),
        "shield": False,
        "runtime": runtime,
    }

    deferred, _hidden = rollout._policy_behavior_action(
        object(), None, visible, commit_runtime=False, **options,
    )
    assert deferred.move_x == 1
    assert runtime.commits == []

    immediate, _hidden = rollout._policy_behavior_action(
        object(), None, visible, **options,
    )
    assert immediate.move_x == 1
    assert runtime.commits == [(immediate, 3)]


def empty_factory(seed: int) -> STGEnvironment:
    return STGEnvironment(
        EmptyScenario(),
        seed=seed,
        config=SimulationConfig(reaction_frames=0, semantic_width=8, semantic_height=8),
    )


def hazard_factory(seed: int) -> STGEnvironment:
    return STGEnvironment(
        RightHandHazard(duration_frames=4),
        seed=seed,
        config=SimulationConfig(
            bounds=Bounds(-64.0, 64.0, -64.0, 64.0),
            player_start=(0.0, 0.0),
            reaction_frames=0,
            semantic_width=8,
            semantic_height=8,
        ),
    )


def test_collect_save_load_and_multi_seed_planner_metrics(tmp_path) -> None:
    output = tmp_path / "teacher.npz"
    demonstrations, episodes = collect_demonstrations(
        empty_factory,
        (11, 12),
        planner=FixedPlanner(),
        vision_config=VISION,
        config=RolloutConfig(decision_interval=2),
        output=output,
    )
    assert demonstrations.global_frames.shape == (6, 2, 6, 8, 8)
    assert demonstrations.local_frames.shape == (6, 2, 6, 8, 8)
    assert np.all(demonstrations.actions == Action().discrete)
    assert all(episode.survived for episode in episodes)
    loaded = load_demonstrations(output)
    np.testing.assert_array_equal(loaded.actions, demonstrations.actions)
    np.testing.assert_array_equal(loaded.episode_ids, demonstrations.episode_ids)
    assert loaded.global_frames.dtype == np.float16

    repeated = evaluate_planner(
        empty_factory,
        (11, 12, 13),
        planner=FixedPlanner(),
        vision_config=VISION,
        config=RolloutConfig(decision_interval=2),
    )
    assert [episode.seed for episode in repeated] == [11, 12, 13]
    assert all(episode.survived and episode.action_agreement == 1.0 for episode in repeated)


def test_dagger_labels_policy_trajectory_with_teacher_actions() -> None:
    torch = pytest.importorskip("torch")

    class WrongPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(memory_size=4, inference_mode="window")
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, global_frames, local_frames, memory=None, hidden=None):
            batch, steps = global_frames.shape[:2]
            logits = torch.zeros((batch, steps, 18), device=global_frames.device)
            logits[..., Action(move_x=1).discrete] = 1.0
            risk = torch.zeros((batch, steps), device=global_frames.device)
            return logits, risk, None

    demonstrations, episodes = collect_dagger_demonstrations(
        WrongPolicy(),
        empty_factory,
        (17,),
        planner=FixedPlanner(),
        vision_config=VISION,
        config=RolloutConfig(decision_interval=2),
        device="cpu",
        shield=False,
    )

    assert np.all(demonstrations.actions == Action().discrete)
    assert demonstrations.risks[-1, -1] > 0.0
    assert episodes[0].survived
    assert episodes[0].action_agreement == 0.0
    assert episodes[0].teacher_overrides == 3


def test_teacher_agreement_and_collision_shield() -> None:
    torch = pytest.importorskip("torch")

    class ConstantPolicy(torch.nn.Module):
        def __init__(self, preferred: int, fallback: int = 4) -> None:
            super().__init__()
            self.config = SimpleNamespace(memory_size=4)
            values = torch.zeros(18)
            values[fallback] = 5.0
            values[preferred] = 6.0
            self.register_buffer("values", values)

        def forward(self, global_frames, local_frames, memory=None):
            batch, steps = global_frames.shape[:2]
            logits = self.values.view(1, 1, -1).expand(batch, steps, -1)
            risk = torch.zeros((batch, steps), device=global_frames.device)
            hidden = torch.zeros((1, batch, 1), device=global_frames.device)
            return logits, risk, hidden

    demonstrations, _ = collect_demonstrations(
        empty_factory,
        (1,),
        planner=FixedPlanner(),
        vision_config=VISION,
        config=RolloutConfig(decision_interval=2),
    )
    assert teacher_action_agreement(ConstantPolicy(Action().discrete), demonstrations) == 1.0

    moving_right = Action(move_x=1).discrete
    evaluation = evaluate_policy_variants(
        ConstantPolicy(moving_right),
        hazard_factory,
        (5,),
        vision_config=VISION,
        config=RolloutConfig(decision_interval=1, shield_horizon=4),
        device="cpu",
    )
    assert not evaluation.raw[0].survived
    assert evaluation.shielded[0].survived
    assert evaluation.raw_survival == 0.0
    assert evaluation.shielded_survival == 1.0
    assert evaluation.raw[0].action_agreement is None


def test_policy_inference_mode_controls_window_and_hidden_lifecycle() -> None:
    torch = pytest.importorskip("torch")

    class RecordingPolicy(torch.nn.Module):
        def __init__(self, inference_mode: str) -> None:
            super().__init__()
            self.config = SimpleNamespace(memory_size=4, inference_mode=inference_mode)
            self.anchor = torch.nn.Parameter(torch.zeros(()))
            self.steps = []
            self.hidden_was_none = []

        def forward(self, global_frames, local_frames, memory=None, hidden=None):
            batch, steps = global_frames.shape[:2]
            self.steps.append(steps)
            self.hidden_was_none.append(hidden is None)
            logits = torch.zeros((batch, steps, 18), device=global_frames.device)
            logits[..., Action().discrete] = 1.0
            risk = torch.zeros((batch, steps), device=global_frames.device)
            next_hidden = torch.ones((1, batch, 1), device=global_frames.device)
            return logits, risk, next_hidden

    window = RecordingPolicy("window")
    evaluate_policy(
        window,
        empty_factory,
        (1,),
        vision_config=VISION,
        config=RolloutConfig(decision_interval=2),
        device="cpu",
    )
    assert window.steps == [2, 2, 2]
    assert window.hidden_was_none == [True, True, True]

    stream = RecordingPolicy("stream")
    evaluate_policy(
        stream,
        empty_factory,
        (1,),
        vision_config=VISION,
        config=RolloutConfig(decision_interval=2),
        device="cpu",
    )
    assert stream.steps == [1, 1, 1]
    assert stream.hidden_was_none == [True, False, False]


def test_policy_memory_provider_is_reset_once_per_seed_before_first_call() -> None:
    torch = pytest.importorskip("torch")

    class ConstantPolicy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(memory_size=4, inference_mode="window")
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def forward(self, global_frames, local_frames, memory=None, hidden=None):
            batch, steps = global_frames.shape[:2]
            logits = torch.zeros((batch, steps, 18), device=global_frames.device)
            logits[..., Action().discrete] = 1.0
            return logits, torch.zeros((batch, steps), device=global_frames.device), None

    class RecordingProvider:
        def __init__(self) -> None:
            self.resets = 0
            self.calls_since_reset = -1
            self.first_call_source_frames = []

        def reset(self, scenario_key, visible) -> None:
            assert scenario_key == "empty"
            self.resets += 1
            self.calls_since_reset = 0
            assert visible.global_frames.shape[0] == VISION.history

        def __call__(self, scenario_key, visible):
            assert scenario_key == "empty"
            assert self.calls_since_reset >= 0
            if self.calls_since_reset == 0:
                self.first_call_source_frames.append(visible.source_frame)
            self.calls_since_reset += 1
            return np.zeros(4, dtype=np.float32)

    provider = RecordingProvider()
    evaluate_policy(
        ConstantPolicy(),
        empty_factory,
        (31, 32, 33),
        vision_config=VISION,
        config=RolloutConfig(decision_interval=2),
        device="cpu",
        memory_provider=provider,
    )
    assert provider.resets == 3
    assert provider.first_call_source_frames == [-1, -1, -1]


def test_planner_teacher_shield_follows_nearest_safe_waypoint() -> None:
    environment = hazard_factory(9)
    move_right = Action(move_x=1)
    plan = SimpleNamespace(
        first_action=move_right,
        steps=(
            SimpleNamespace(position=(0.0, 0.0)),
            SimpleNamespace(position=(8.0, 0.0)),
        ),
    )
    safe = imminent_safe_actions(environment, 2)
    assert move_right.discrete not in safe
    selected = planner_teacher_action(environment, plan, hold_frames=2)
    assert selected == Action(move_x=1, slow=True)


def test_planner_teacher_shield_does_not_replace_an_already_safe_action() -> None:
    environment = empty_factory(9)
    move_right = Action(move_x=1)
    plan = SimpleNamespace(
        first_action=move_right,
        steps=(
            SimpleNamespace(position=(0.0, -176.0)),
            SimpleNamespace(position=(-40.0, -176.0)),
        ),
    )
    assert planner_teacher_action(environment, plan, hold_frames=2) == move_right


def test_toward_policy_shield_chooses_nearest_safe_preferred_endpoint() -> None:
    environment = hazard_factory(9)
    preferred = Action(move_x=1)
    horizon = 4
    endpoints = _safe_action_endpoints(environment, horizon)
    target = _candidate_player_paths(environment, preferred, horizon)[-1]

    selected = shield_action_toward(environment, preferred, horizon=horizon)

    assert preferred.discrete not in endpoints
    assert selected.discrete in endpoints
    selected_distance = (
        (endpoints[selected.discrete][0] - target[0]) ** 2
        + (endpoints[selected.discrete][1] - target[1]) ** 2
    )
    assert selected_distance == min(
        (x - target[0]) ** 2 + (y - target[1]) ** 2
        for x, y in endpoints.values()
    )


def test_batched_shield_matches_independent_clone_reference() -> None:
    environment = make_environment(
        "stage5_boss4",
        seed=71,
        duration_frames=180,
        config=SimulationConfig(reaction_frames=0, action_hold_frames=3),
    )
    for _ in range(150):
        environment._advance(Action(), build_semantic=False, detect_collision=False)
    batched = _safe_action_endpoints(environment, 3)

    reference = {}
    for discrete in range(18):
        clone = environment.clone()
        action = Action.from_discrete(discrete)
        for _ in range(3):
            result = clone._advance(action, build_semantic=False, detect_collision=True)
            if result.outcome.value == "hit":
                break
        else:
            reference[discrete] = (clone.player.x, clone.player.y)
    assert batched == reference


def test_external_memory_changes_second_attempt() -> None:
    first_action = Action(move_x=1)

    def first_controller(_environment, _vision, _plan, _memory):
        return first_action

    def second_controller(_environment, _vision, _plan, memory):
        assert memory is not None
        return memory.route[0]

    with EpisodicMemory() as memory:
        result = benchmark_second_attempt(
            hazard_factory,
            7,
            first_controller=first_controller,
            second_controller=second_controller,
            route_builder=lambda _trace: (Action(),),
            planner=FixedPlanner(),
            memory_store=memory,
            vision_config=VISION,
            config=RolloutConfig(decision_interval=1, shield_horizon=4),
            trigger_lead=2,
        )
        assert len(memory) == 1
        assert result.memory.successes == 1
    assert not result.first.survived
    assert result.second.survived
    assert result.passed
