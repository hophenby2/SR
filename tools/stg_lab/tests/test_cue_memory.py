from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from stg_lab.cue_memory import StatefulCueMemoryProvider, cue_condition_demonstrations
from stg_lab.training import Demonstrations
from stg_lab.vision import VisionObservation


ARTIFACTS = Path(__file__).parents[1] / "artifacts"


def _visible(motion: float, *, source_frame: int = 0) -> VisionObservation:
    global_frames = np.zeros((4, 6, 56, 48), dtype=np.float32)
    local_frames = np.zeros((4, 6, 40, 40), dtype=np.float32)
    world_y = np.linspace(-224.0, 224.0, 56)
    rows = (world_y >= 80.0) & (world_y <= 144.0)
    global_frames[-1, 0, rows] = 0.5
    global_frames[-1, 1, rows] = motion
    return VisionObservation(global_frames, local_frames, source_frame)


def _demonstrations(motions: list[float], episode_ids: list[int]) -> Demonstrations:
    observations = [_visible(motion, source_frame=index) for index, motion in enumerate(motions)]
    samples = len(observations)
    memory = np.zeros((samples, 4, 4), dtype=np.float32)
    memory[..., 0] = 1.0
    return Demonstrations(
        global_frames=np.stack([item.global_frames for item in observations]),
        local_frames=np.stack([item.local_frames for item in observations]),
        actions=np.zeros((samples, 4), dtype=np.int64),
        risks=np.zeros((samples, 4), dtype=np.float32),
        memory=memory,
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
    )


def test_phase_cue_is_zero_until_visible_then_freezes_sign_and_amplitude() -> None:
    provider = StatefulCueMemoryProvider()
    provider.reset("stage5_boss3:lunatic", _visible(0.0))

    assert provider("stage5_boss3:lunatic", _visible(0.0)).tolist() == [1.0, 0.0, 0.0, 0.0]
    captured = provider("stage5_boss3:lunatic", _visible(-0.37))
    assert captured.tolist() == pytest.approx([1.0, 0.0, -1.0, 0.37])
    assert provider("stage5_boss3:lunatic", _visible(0.91)).tolist() == pytest.approx(captured)


def test_different_opening_phases_retain_direction_and_amplitude() -> None:
    negative = StatefulCueMemoryProvider()
    negative.reset("stage5_boss3", _visible(0.0))
    negative_value = negative("stage5_boss3", _visible(-0.21))

    positive = StatefulCueMemoryProvider()
    positive.reset("stage5_boss3", _visible(0.0))
    positive_value = positive("stage5_boss3", _visible(0.70))

    assert negative_value[2] == -1.0
    assert positive_value[2] == 1.0
    assert negative_value[3] == pytest.approx(0.21)
    assert positive_value[3] == pytest.approx(0.70)


def test_boss4_identity_is_unchanged_even_if_roi_moves() -> None:
    provider = StatefulCueMemoryProvider()
    provider.reset("stage5_boss4:lunatic", _visible(0.0))
    assert provider("stage5_boss4:lunatic", _visible(0.8)).tolist() == [0.0, 1.0, 0.0, 0.0]
    assert not provider.captured


def test_archive_conversion_matches_online_provider_per_sample_without_backfill() -> None:
    demonstrations = _demonstrations(
        [0.0, 0.0, 0.37, 0.08, 0.0, -0.64, 0.08],
        [41, 41, 41, 41, 99, 99, 99],
    )
    demonstrations.correction_mask[2, -1] = True
    converted = cue_condition_demonstrations(
        demonstrations,
        scenario_by_episode={41: "stage5_boss3", 99: "stage5_boss3"},
    )

    provider = StatefulCueMemoryProvider()
    online: list[np.ndarray] = []
    current = None
    for sample, episode_id in enumerate(demonstrations.episode_ids):
        visible = VisionObservation(
            demonstrations.global_frames[sample],
            demonstrations.local_frames[sample],
            source_frame=10000 + sample,
        )
        if int(episode_id) != current:
            current = int(episode_id)
            provider.reset("stage5_boss3", visible)
        online.append(provider("stage5_boss3", visible))

    assert np.array_equal(converted.episode_ids, demonstrations.episode_ids)
    assert np.array_equal(converted.correction_mask, demonstrations.correction_mask)
    assert np.array_equal(converted.memory[:, -1], np.stack(online))
    assert np.all(converted.memory[:2, :, 2:] == 0.0)
    assert np.all(converted.memory[4, :, 2:] == 0.0)
    assert np.all(converted.memory[2:, :, 2] == np.asarray([1, 1, 0, -1, -1])[:, None])


def test_archive_conversion_rejects_noncontiguous_episode_blocks() -> None:
    demonstrations = _demonstrations([0.0, 0.3, 0.2], [1, 2, 1])
    with pytest.raises(ValueError, match="contiguous"):
        cue_condition_demonstrations(demonstrations, scenario_by_episode="stage5_boss3")


def test_actual_boss3_archives_match_online_capture_and_known_phase_values() -> None:
    expected = (0.369384765625, -0.213134765625, -0.69775390625, -1.0)
    observed: list[float] = []
    for archive in ("boss3_train_a.npz", "boss3_train_b.npz"):
        demonstrations = Demonstrations.load(ARTIFACTS / archive)
        converted = cue_condition_demonstrations(
            demonstrations,
            scenario_by_episode="stage5_boss3:lunatic",
        )
        for episode_id in np.unique(demonstrations.episode_ids):
            indices = np.flatnonzero(demonstrations.episode_ids == episode_id)
            provider = StatefulCueMemoryProvider()
            online = []
            for position, sample in enumerate(indices):
                visible = VisionObservation(
                    demonstrations.global_frames[sample],
                    demonstrations.local_frames[sample],
                    source_frame=9999 - position,
                )
                if position == 0:
                    provider.reset("stage5_boss3:lunatic", visible)
                online.append(provider("stage5_boss3:lunatic", visible))
            online_array = np.stack(online)
            np.testing.assert_array_equal(converted.memory[indices, -1], online_array)
            np.testing.assert_array_equal(online_array[:2, 2:], 0.0)
            observed.append(float(online_array[2, 2] * online_array[2, 3]))

    assert observed == pytest.approx(expected)


def test_actual_boss4_archive_never_acquires_boss3_cue() -> None:
    demonstrations = Demonstrations.load(ARTIFACTS / "boss4_canonical_heldout.npz")
    converted = cue_condition_demonstrations(
        demonstrations,
        scenario_by_episode="stage5_boss4:lunatic",
    )
    np.testing.assert_array_equal(converted.memory[..., :2], demonstrations.memory[..., :2])
    np.testing.assert_array_equal(converted.memory[..., 2:], 0.0)

    provider = StatefulCueMemoryProvider()
    for episode_id in np.unique(demonstrations.episode_ids):
        indices = np.flatnonzero(demonstrations.episode_ids == episode_id)
        for position, sample in enumerate(indices):
            visible = VisionObservation(
                demonstrations.global_frames[sample],
                demonstrations.local_frames[sample],
                source_frame=-100 - position,
            )
            if position == 0:
                provider.reset("stage5_boss4:lunatic", visible)
            assert provider("stage5_boss4:lunatic", visible).tolist() == [0.0, 1.0, 0.0, 0.0]
