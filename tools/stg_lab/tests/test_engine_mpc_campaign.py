from __future__ import annotations

import json
from typing import Any, Mapping

import pytest

from stg_lab.engine_mpc import (
    EngineMPC,
    MPCConfig,
    MPCDecision,
    RegionDynamicsMemory,
)
from stg_lab.engine_mpc_campaign import (
    EngineMPCCampaignConfig,
    run_engine_mpc_campaign,
)
from stg_lab.engine_runtime import local_runtime_source_fingerprints
from stg_lab.protocol import Action


def _resources() -> dict[str, int]:
    return {
        "lifeleft": 7,
        "bomb": 3,
        "power": 0,
        "faith": 0,
        "score": 0,
    }


class FakeCampaignClient:
    def __init__(
        self,
        *,
        missing_active_stage: int | None = None,
        final_death: int = 0,
        completion_reason: str = "campaign_complete",
        frames_per_stage: int = 2,
    ) -> None:
        self.missing_active_stage = missing_active_stage
        self.final_death = final_death
        self.completion_reason = completion_reason
        self.frames_per_stage = frames_per_stage
        self.frame = 0
        self.stage_index = 1
        self.transition_count = 0
        self.stage_active_content_seen = False
        self.reset_calls = 0
        self.actions: list[Action] = []
        self.overlay_states: list[Mapping[str, Any] | None] = []
        self.runtime_source_crc32 = local_runtime_source_fingerprints()[0]

    @staticmethod
    def _stage_name(index: int) -> str:
        return f"Stage {index}@Lunatic"

    def ping(self) -> dict[str, Any]:
        return {
            "protocol": 2,
            "commands": [
                "ping", "catalog", "reset_campaign", "step", "display",
            ],
            "session_id": "fake-campaign-session",
            "process_nonce": "fake-campaign-process",
            "runtime_identity": {
                "process_id": 42,
                "source_crc32": self.runtime_source_crc32,
            },
        }

    def catalog(self) -> dict[str, Any]:
        return {"catalog": {"stages": [
            {
                "episode_kind": "stage",
                "stage": self._stage_name(index),
                "stage_index": index,
                "difficulty": "Lunatic",
                "completion_reason": "stage_complete",
            }
            for index in range(1, 6)
        ]}}

    def reset_campaign(
        self,
        difficulty: str,
        *,
        seed: int,
        player: str,
        options: Mapping[str, Any],
    ) -> dict[str, Any]:
        assert difficulty == "Lunatic"
        assert options == {}
        self.reset_calls += 1
        self.frame = 0
        self.stage_index = 1
        self.transition_count = 0
        self.stage_active_content_seen = False
        return {
            "reset": {
                "episode_kind": "campaign",
                "difficulty": difficulty,
                "stage_index": 1,
                "stage_name": self._stage_name(1),
                "stage_count": 5,
                "seed": seed,
                "player": player,
            },
            "observation": self._observation(),
        }

    def set_rendering(self, enabled: bool, *, every: int = 1) -> dict[str, Any]:
        return {"render": enabled, "every": every}

    def _active_for_stage(self, index: int) -> bool:
        return index != self.missing_active_stage

    def _completed_stages(self) -> list[dict[str, Any]]:
        count = min(self.transition_count, 5)
        return [
            {
                "stage_index": index,
                "stage_name": self._stage_name(index),
                "completion_episode_frame": index * self.frames_per_stage,
                "active_content_seen": self._active_for_stage(index),
                "resources": _resources(),
                "hidden_route": False,
            }
            for index in range(1, count + 1)
        ]

    def _transitions(self) -> list[dict[str, Any]]:
        return [
            {
                "from_stage_index": index,
                "from_stage_name": self._stage_name(index),
                "to_stage_index": 0 if index == 5 else index + 1,
                "to_stage_name": "menu" if index == 5 else self._stage_name(index + 1),
                "episode_frame": index * self.frames_per_stage,
                "active_content_seen": self._active_for_stage(index),
                "resources": _resources(),
                "hidden_route": False,
            }
            for index in range(1, self.transition_count + 1)
        ]

    def _observation(self) -> dict[str, Any]:
        complete = self.transition_count == 5
        completed = self._completed_stages()
        all_completed_active = bool(completed) and all(
            value["active_content_seen"] for value in completed
        )
        return {
            "episode_frame": self.frame,
            "terminated": complete,
            "termination_reason": self.completion_reason if complete else None,
            "performance": {"native_fps": 60.0, "object_count": 20},
            "stage": {
                "scenario": self._stage_name(self.stage_index),
                "stage_index": self.stage_index,
            },
            "campaign": {
                "schema_version": 1,
                "difficulty": "Lunatic",
                "stage_index": self.stage_index,
                "stage_name": self._stage_name(self.stage_index),
                "stage_count": 5,
                "stages_completed": len(completed),
                "completed_stages": completed,
                "active_content_seen": (
                    all_completed_active or self.stage_active_content_seen
                ),
                "stage_active_content_seen": self.stage_active_content_seen,
                "stage_transition_count": self.transition_count,
                "transitions": self._transitions(),
                "initial_resources": _resources(),
                "resources": _resources(),
                "initial_hidden_route": False,
                "hidden_route": False,
                "campaign_complete": complete,
            },
            "world": {"pl": -192.0, "pr": 192.0, "pb": -224.0, "pt": 224.0},
            "player": {
                "x": 0.0,
                "y": -176.0,
                "a": 0.5,
                "b": 0.5,
                "hspeed": 4.0,
                "lspeed": 2.0,
                "death": self.final_death if complete else 0,
                "protect": 0,
                "status": "normal",
            },
            "enemy_bullets": [],
            "enemies": [{
                "id": self.stage_index * 100,
                "x": 0.0,
                "y": 120.0,
                "a": 16.0,
                "b": 16.0,
                "hp": 100.0,
                "maxhp": 100.0,
                "collidable": False,
            }],
            "nontjt_enemies": [],
            "indestructibles": [],
            "lasers": [],
        }

    def step(
        self,
        action: Action,
        *,
        repeat: int = 1,
        controller_overlay_state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        assert repeat == 1
        self.actions.append(action)
        self.overlay_states.append(controller_overlay_state)
        self.frame += 1
        self.stage_active_content_seen = self._active_for_stage(self.stage_index)
        if self.frame % self.frames_per_stage == 0:
            self.transition_count += 1
            if self.transition_count < 5:
                self.stage_index += 1
                self.stage_active_content_seen = False
        return {"observation": self._observation()}


class BoundaryCountingMPC(EngineMPC):
    def __init__(self, config: MPCConfig) -> None:
        self.boundary_calls = 0
        super().__init__(config)

    def on_stage_boundary(self) -> None:
        self.boundary_calls += 1
        super().on_stage_boundary()


class CampaignCadenceMPC(BoundaryCountingMPC):
    def __init__(self, config: MPCConfig) -> None:
        self.observed_frames: list[tuple[int, int, int]] = []
        self.selected_frames: list[tuple[int, int, int]] = []
        super().__init__(config)

    @staticmethod
    def _frames(observed: Mapping[str, Any]) -> tuple[int, int, int]:
        stage = observed.get("stage")
        assert isinstance(stage, Mapping)
        stage_index = stage.get("stage_index")
        source_frame = observed.get("episode_frame")
        current_frame = observed.get("own_player_observation_frame")
        assert isinstance(stage_index, int) and not isinstance(stage_index, bool)
        assert isinstance(source_frame, int) and not isinstance(source_frame, bool)
        assert isinstance(current_frame, int) and not isinstance(current_frame, bool)
        return stage_index, source_frame, current_frame

    def observe(self, observed: Mapping[str, Any]) -> int:
        self.observed_frames.append(self._frames(observed))
        return super().observe(observed)

    def select(self, observed: Mapping[str, Any]) -> MPCDecision:
        self.selected_frames.append(self._frames(observed))
        return super().select(observed)


def _controller() -> BoundaryCountingMPC:
    return BoundaryCountingMPC(MPCConfig(
        observation_delay=0,
        horizon_frames=36,
        beam_width=8,
        region_beam_width=16,
    ))


def _run(client: FakeCampaignClient, controller: EngineMPC | None = None) -> dict[str, Any]:
    return run_engine_mpc_campaign(
        client,  # type: ignore[arg-type]
        difficulty="Lunatic",
        seed=42,
        player="reimu_player",
        controller=controller or _controller(),
        config=EngineMPCCampaignConfig(
            max_frames=20,
            observation_delay=0,
        ),
    )


def test_continuous_campaign_uses_one_reset_and_clears_four_boundaries() -> None:
    client = FakeCampaignClient()
    controller = _controller()

    report = _run(client, controller)

    assert report["passed"] is True
    assert report["termination_reason"] == "campaign_complete"
    assert report["reset_count"] == 1
    assert client.reset_calls == 1
    assert report["stage_boundary_count"] == 4
    assert controller.boundary_calls == 4
    assert report["campaign_complete_evidence"] is True
    assert report["all_stages_active_content_seen"] is True
    assert report["outcome_evidence"]["final_player"]["death"] == 0
    assert report["all_control_sources_live_mpc"] is True
    assert report["continuous_fire"] is True
    assert report["shoot_frames"] == report["frames"] == len(client.actions)
    assert all(action.shoot and not action.spell for action in client.actions)
    assert all(
        value["control_source"] == "live_mpc"
        for value in report["decisions"]
    )
    assert all(
        value["reporting_only_authority_player"] is not None
        and value["planned_actions"]
        for value in report["decisions"]
    )
    assert report["external_memory_free"] is True
    assert all(value is None for value in report["external_memory"].values())
    assert report["controller"]["config"]["region_dynamics_memory"] is None
    assert report["native_replay"] is None
    assert report["replay_supported"] is False
    terminal_window = report["terminal_observation_window"]
    assert terminal_window["output_only_not_reused_by_controller"] is True
    assert terminal_window["authority_observations_are_controller_input"] is False
    assert terminal_window["controller_inputs_are_historical_live_inputs"] is True
    assert terminal_window["stage_index"] == 5
    assert 1 <= len(terminal_window["observations"]) <= 24
    assert 1 <= len(terminal_window["controller_inputs"]) <= 8
    assert all(
        "campaign" not in value["observation"]
        for value in terminal_window["controller_inputs"]
    )
    json.dumps(report, allow_nan=False)


def test_campaign_observation_cadence_resets_cleanly_at_stage_boundaries() -> None:
    client = FakeCampaignClient(frames_per_stage=9)
    controller = CampaignCadenceMPC(MPCConfig(
        observation_delay=2,
        horizon_frames=36,
        beam_width=8,
        region_beam_width=16,
    ))

    report = run_engine_mpc_campaign(
        client,  # type: ignore[arg-type]
        difficulty="Lunatic",
        seed=42,
        player="reimu_player",
        controller=controller,
        config=EngineMPCCampaignConfig(max_frames=50, observation_delay=2),
    )

    assert report["passed"] is True
    assert [value["start_episode_frame"] for value in report["decisions"]] == list(
        range(0, 45, 3)
    )
    assert [value[2] for value in controller.selected_frames] == list(
        range(0, 45, 3)
    )
    for stage_index in range(1, 6):
        stage_start = (stage_index - 1) * 9
        assert [
            source_frame
            for observed_stage, source_frame, _ in controller.observed_frames
            if observed_stage == stage_index
        ] == list(range(stage_start, stage_start + 7))
        assert [
            source_frame
            for selected_stage, source_frame, _ in controller.selected_frames
            if selected_stage == stage_index
        ] == [stage_start, stage_start + 1, stage_start + 4]


@pytest.mark.parametrize(
    ("client", "failed_field"),
    [
        (FakeCampaignClient(missing_active_stage=3), "all_stages_active_content_seen"),
        (FakeCampaignClient(final_death=100), "passed"),
        (FakeCampaignClient(completion_reason="stage_complete"), "passed"),
    ],
)
def test_campaign_strict_outcome_failures(
    client: FakeCampaignClient,
    failed_field: str,
) -> None:
    report = _run(client)

    assert report["passed"] is False
    assert report[failed_field] is False


def test_campaign_rejects_controller_region_memory_before_reset() -> None:
    client = FakeCampaignClient()
    memory = RegionDynamicsMemory(
        minimum_radius=7.0,
        maximum_radius=28.0,
        growth_rate=0.7,
        contraction_rate=0.7,
        expanding_frames=30.0,
        maximum_hold_frames=30.0,
        contracting_frames=30.0,
        minimum_hold_frames=90.0,
        cycle_frames=180.0,
    )
    controller = EngineMPC(MPCConfig(
        observation_delay=0,
        horizon_frames=36,
        region_dynamics_memory=memory,
    ))

    with pytest.raises(ValueError, match="must not contain region dynamics memory"):
        _run(client, controller)
    assert client.reset_calls == 0


def test_campaign_rejects_python_source_drift_during_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import stg_lab.engine_mpc_campaign as campaign_module

    hashes = iter(("a" * 64, "b" * 64))
    monkeypatch.setattr(
        campaign_module,
        "source_tree_sha256",
        lambda: next(hashes),
    )

    report = _run(FakeCampaignClient())

    assert report["implementation_sha256"] == "a" * 64
    assert report["implementation_sha256_end"] == "b" * 64
    assert report["implementation_source_unchanged"] is False
    assert report["passed"] is False


def test_stage_boundary_clears_transient_state_without_replacing_config() -> None:
    controller = _controller()
    config = controller.config
    controller._last_source_frame = 123
    controller._last_decision_frame = 123
    controller._committed_plan = (Action(move_x=1),)
    controller._committed_plan_is_region = True
    controller._committed_plan_is_gap = True
    controller._active_gap_key = "gap"
    controller._region_topology.target_x = 50.0

    controller.on_stage_boundary()

    assert controller.config is config
    assert controller._last_source_frame is None
    assert controller._last_decision_frame is None
    assert controller._committed_plan == ()
    assert controller._committed_plan_is_region is False
    assert controller._committed_plan_is_gap is False
    assert controller._active_gap_key is None
    assert controller._region_topology.target_x is None
