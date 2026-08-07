from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest

pytest.importorskip("torch")

import experiments.compare_plain_certified_set_cv30 as cv30
from experiments.compare_plain_certified_set_cv30 import (
    ALL_TRAINING_SEEDS,
    EXPANSION_ACQUISITION_COHORT,
    EXPANSION_INVENTORY_CONTRACT,
    EXPANSION_TRAINING_SEEDS,
    LEGACY_ACQUISITION_COHORT,
    LEGACY_TRAINING_SEEDS,
    PLAIN_ARM_NAME,
    PLAIN_OBJECTIVE_CONFIG,
    PROHIBITED_SOURCE_SEEDS,
    _fixed_cv30_folds,
    _merge_verified_sources,
    _plain_promotion_gate,
    _plain_summary,
    _select_expansion_inventory,
    _split_acquisition_audit,
    _verify_expansion_sources,
)
from experiments.compare_preferred_objectives_cv import _fixed_folds

CHECKPOINT_SHA256 = "a" * 64


class ProtocolOnly(Mapping[str, object]):
    """Record that raises if selection touches any path-bearing field."""

    def __init__(self, seed: int, **overrides: object) -> None:
        self.values = {
            "seed": seed,
            "role": "training",
            "strict_prefix_zero": True,
            "strict_success": True,
            "shoot_command_rate": 1.0,
            "external_region_memory": None,
            **overrides,
        }

    def __getitem__(self, key: str) -> object:
        if key in {
            "dataset",
            "dataset_sha256",
            "report",
            "report_sha256",
            "manifest",
            "manifest_sha256",
        }:
            raise AssertionError(f"path-bearing field was accessed: {key}")
        return self.values[key]

    def __iter__(self) -> Iterator[str]:
        yield from self.values

    def __len__(self) -> int:
        return len(self.values)


def expansion_inventory(
    *,
    records: list[Mapping[str, object]] | None = None,
    **overrides: object,
) -> dict[str, object]:
    return {
        **EXPANSION_INVENTORY_CONTRACT,
        "seeds": list(EXPANSION_TRAINING_SEEDS),
        "reserved_seeds_excluded": sorted(PROHIBITED_SOURCE_SEEDS),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_inventory": records
        if records is not None
        else [ProtocolOnly(seed) for seed in EXPANSION_TRAINING_SEEDS],
        **overrides,
    }


def test_cv30_folds_preserve_cv15_roles_and_stratify_acquisition() -> None:
    legacy_folds = _fixed_folds()
    folds = _fixed_cv30_folds()
    assert len(folds) == 3
    audit_seeds: list[int] = []
    for legacy, fold in zip(legacy_folds, folds, strict=True):
        for name, expected_size, expected_per_cohort in (
            ("fit_seeds", 16, 8),
            ("calibration_seeds", 4, 2),
            ("audit_seeds", 10, 5),
        ):
            seeds = getattr(fold, name)
            assert len(seeds) == expected_size
            acquisition = _split_acquisition_audit(seeds)
            assert acquisition["strictly_interleaved"] is True
            assert acquisition["counts"] == {
                LEGACY_ACQUISITION_COHORT: expected_per_cohort,
                EXPANSION_ACQUISITION_COHORT: expected_per_cohort,
            }
            assert tuple(seeds[0::2]) == getattr(legacy, name)
        fit = set(fold.fit_seeds)
        calibration = set(fold.calibration_seeds)
        audit = set(fold.audit_seeds)
        assert not fit & calibration
        assert not fit & audit
        assert not calibration & audit
        assert fit | calibration | audit == set(ALL_TRAINING_SEEDS)
        audit_seeds.extend(fold.audit_seeds)
    assert sorted(audit_seeds) == sorted(ALL_TRAINING_SEEDS)
    assert len(audit_seeds) == len(set(audit_seeds)) == 30


def test_expansion_selection_validates_protocol_before_path_access() -> None:
    selected = _select_expansion_inventory(
        expansion_inventory(),
        checkpoint_sha256=CHECKPOINT_SHA256,
    )
    assert [record["seed"] for record in selected] == list(EXPANSION_TRAINING_SEEDS)


@pytest.mark.parametrize(
    ("override", "message"),
    (
        ({"intervene_on_disagreement": False}, "intervene_on_disagreement"),
        ({"intervene_on_disagreement": 1}, "intervene_on_disagreement"),
        ({"schema_version": True}, "schema_version"),
        ({"checkpoint_sha256": "b" * 64}, "checkpoint SHA-256"),
        ({"seeds": list(reversed(EXPANSION_TRAINING_SEEDS))}, "fixed ordered cohort"),
        ({"reserved_seeds_excluded": []}, "reserved seed declaration"),
    ),
)
def test_expansion_header_is_fixed(override: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _select_expansion_inventory(
            expansion_inventory(**override),
            checkpoint_sha256=CHECKPOINT_SHA256,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("strict_prefix_zero", False),
        ("strict_prefix_zero", 1),
        ("strict_success", False),
        ("strict_success", 1),
        ("shoot_command_rate", 0.999),
        ("shoot_command_rate", 1),
        ("shoot_command_rate", True),
        ("external_region_memory", "memory.json"),
    ),
)
def test_expansion_episode_contract_is_fixed(field: str, value: object) -> None:
    records = [ProtocolOnly(seed) for seed in EXPANSION_TRAINING_SEEDS]
    records[4] = ProtocolOnly(EXPANSION_TRAINING_SEEDS[4], **{field: value})
    with pytest.raises(ValueError, match=field):
        _select_expansion_inventory(
            expansion_inventory(records=records),
            checkpoint_sha256=CHECKPOINT_SHA256,
        )


def test_expansion_source_order_and_whitelist_are_fail_closed() -> None:
    reordered = [ProtocolOnly(seed) for seed in EXPANSION_TRAINING_SEEDS]
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ValueError, match="seed order"):
        _select_expansion_inventory(
            expansion_inventory(records=reordered),
            checkpoint_sha256=CHECKPOINT_SHA256,
        )

    unexpected = [ProtocolOnly(seed) for seed in EXPANSION_TRAINING_SEEDS]
    unexpected.append(ProtocolOnly(10326))
    with pytest.raises(ValueError, match="unexpected expansion source"):
        _select_expansion_inventory(
            expansion_inventory(records=unexpected),
            checkpoint_sha256=CHECKPOINT_SHA256,
        )

    unexpected[-1] = ProtocolOnly(10326, role="validation")
    with pytest.raises(ValueError, match="unexpected expansion source"):
        _select_expansion_inventory(
            expansion_inventory(records=unexpected),
            checkpoint_sha256=CHECKPOINT_SHA256,
        )


def verified_source_record(seed: int) -> dict[str, object]:
    stem = f"seed-{seed}"
    return {
        "seed": seed,
        "role": "training",
        "strict_prefix_zero": True,
        "strict_success": True,
        "dataset": f"{stem}.npz",
        "dataset_sha256": "1" * 64,
        "report": f"{stem}.json",
        "report_sha256": "2" * 64,
        "manifest": f"{stem}.manifest.json",
        "manifest_sha256": "3" * 64,
        "decisions": 1090,
        "teacher_interventions": 980,
        "shoot_command_rate": 1.0,
        "external_region_memory": None,
    }


def test_expansion_verification_rebuilds_every_strict_triplet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [verified_source_record(seed) for seed in EXPANSION_TRAINING_SEEDS]
    calls: list[tuple[int, Path, bool]] = []

    def validate(
        paths: Mapping[str, Path],
        *,
        seed: int,
        checkpoint: Path,
        intervene_on_disagreement: bool,
    ) -> dict[str, object]:
        calls.append((seed, checkpoint, intervene_on_disagreement))
        assert paths["dataset"] == Path(f"seed-{seed}.npz")
        return verified_source_record(seed)

    monkeypatch.setattr(cv30, "_validate_strict_prefix0_triplet", validate)
    triplets, provenance = _verify_expansion_sources(
        records,
        checkpoint=Path("parent.pt"),
    )
    assert len(triplets) == len(provenance) == 15
    assert calls == [
        (seed, Path("parent.pt"), True) for seed in EXPANSION_TRAINING_SEEDS
    ]
    assert all(record["strict_native_triplet_revalidated"] for record in provenance)


def test_expansion_verification_rejects_rebuilt_record_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [verified_source_record(seed) for seed in EXPANSION_TRAINING_SEEDS]
    records[0]["dataset_sha256"] = "f" * 64
    monkeypatch.setattr(
        cv30,
        "_validate_strict_prefix0_triplet",
        lambda paths, *, seed, checkpoint, intervene_on_disagreement: (
            verified_source_record(seed)
        ),
    )
    with pytest.raises(ValueError, match="dataset_sha256 differs"):
        _verify_expansion_sources(records, checkpoint=Path("parent.pt"))


def test_expansion_verification_propagates_native_protocol_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [verified_source_record(seed) for seed in EXPANSION_TRAINING_SEEDS]

    def reject(*args: object, **kwargs: object) -> dict[str, object]:
        raise ValueError("strict report must contain death=0")

    monkeypatch.setattr(cv30, "_validate_strict_prefix0_triplet", reject)
    with pytest.raises(ValueError, match="death=0"):
        _verify_expansion_sources(records, checkpoint=Path("parent.pt"))


def test_merged_sources_are_interleaved_and_labeled_by_cohort() -> None:
    legacy_triplets = [
        (Path(f"old-{seed}"), Path("r"), Path("m")) for seed in LEGACY_TRAINING_SEEDS
    ]
    expansion_triplets = [
        (Path(f"new-{seed}"), Path("r"), Path("m")) for seed in EXPANSION_TRAINING_SEEDS
    ]
    legacy_provenance = [{"seed": seed} for seed in LEGACY_TRAINING_SEEDS]
    expansion_provenance = [
        {"seed": seed, "acquisition_cohort": EXPANSION_ACQUISITION_COHORT}
        for seed in EXPANSION_TRAINING_SEEDS
    ]
    triplets, provenance = _merge_verified_sources(
        legacy_triplets,
        legacy_provenance,
        expansion_triplets,
        expansion_provenance,
    )
    assert [record["seed"] for record in provenance] == list(ALL_TRAINING_SEEDS)
    assert [record["acquisition_cohort"] for record in provenance[0::2]] == [
        LEGACY_ACQUISITION_COHORT
    ] * 15
    assert [record["acquisition_cohort"] for record in provenance[1::2]] == [
        EXPANSION_ACQUISITION_COHORT
    ] * 15
    assert [path.name for path, _, _ in triplets[0::2]] == [
        f"old-{seed}" for seed in LEGACY_TRAINING_SEEDS
    ]


def summary(
    *,
    targets: int = 359,
    finite: int = 359,
    equivalent: int = 59,
    direction: int = 81,
    speed: int = 194,
    calibration_folds: int = 2,
    audit_runtime_eligible_folds: int = 2,
    calibrated_audit_runtime_ineligible_folds: list[int] | None = None,
) -> dict[str, object]:
    return {
        "calibration_successful_folds": calibration_folds,
        "audit_runtime_eligible_folds": audit_runtime_eligible_folds,
        "calibrated_audit_runtime_ineligible_folds": (
            []
            if calibrated_audit_runtime_ineligible_folds is None
            else calibrated_audit_runtime_ineligible_folds
        ),
        "outer_audit_micro": {
            "targets": targets,
            "finite_top1": finite,
            "equivalent_top1": equivalent,
            "direction_correct": direction,
            "speed_correct": speed,
        },
    }


def test_plain_summary_counts_only_safe_calibrated_audit_runtimes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cv30, "_sum_raw", lambda rows: {"targets": len(rows)})

    def fold(index: int, *, calibrated: bool, eligible: object) -> dict[str, object]:
        return {
            "fold": index,
            "plain": {
                "calibration": {"success": calibrated},
                "runtime_metrics": (
                    {"audit": {"offline_deployment_eligible": eligible}}
                    if calibrated
                    else None
                ),
                "raw_action_metrics": {"audit": {}},
            },
        }

    result = _plain_summary(
        [
            fold(0, calibrated=True, eligible=True),
            fold(1, calibrated=True, eligible=False),
            fold(2, calibrated=False, eligible=True),
        ]
    )
    assert result["calibration_successful_folds"] == 2
    assert result["calibration_failed_folds"] == [2]
    assert result["audit_runtime_eligible_folds"] == 1
    assert result["audit_runtime_eligible_fold_indices"] == [0]
    assert result["calibrated_audit_runtime_ineligible_folds"] == [1]


def test_plain_promotion_gate_accepts_exact_preregistered_boundaries() -> None:
    gate = _plain_promotion_gate(summary(), epochs=6)
    assert gate["applicable"] is True
    assert gate["passed"] is True
    assert gate["eligible_for_fixed_e20_followup"] is True
    assert all(gate["checks"].values())


def test_plain_promotion_gate_uses_exact_ratio_for_new_denominator() -> None:
    # 118/718 equals 59/359 exactly, while 117/718 is strictly below it.
    exact = _plain_promotion_gate(
        summary(targets=718, finite=718, equivalent=118, direction=162, speed=388),
        epochs=6,
    )
    below = _plain_promotion_gate(
        summary(targets=718, finite=718, equivalent=117, direction=162, speed=388),
        epochs=6,
    )
    assert exact["passed"] is True
    assert (
        below["checks"]["outer_audit_equivalent_top1_rate_at_least_legacy_plain"]
        is False
    )


@pytest.mark.parametrize(
    "values",
    (
        {"finite": 358},
        {"equivalent": 58},
        {"direction": 80},
        {"speed": 193},
        {"calibration_folds": 1},
        {"audit_runtime_eligible_folds": 1},
        {
            "calibrated_audit_runtime_ineligible_folds": [2],
            "calibration_folds": 3,
            "audit_runtime_eligible_folds": 2,
        },
    ),
)
def test_plain_promotion_gate_fails_each_preregistered_floor(
    values: dict[str, int],
) -> None:
    gate = _plain_promotion_gate(summary(**values), epochs=6)
    assert gate["passed"] is False
    assert gate["eligible_for_fixed_e20_followup"] is False


def test_plain_promotion_gate_is_inapplicable_outside_e6() -> None:
    gate = _plain_promotion_gate(summary(), epochs=20)
    assert gate["applicable"] is False
    assert gate["passed"] is False
    assert gate["eligible_for_fixed_e20_followup"] is False


def test_cv30_declares_only_plain_certified_set_objective() -> None:
    assert PLAIN_ARM_NAME == "equivalence"
    assert PLAIN_OBJECTIVE_CONFIG == {
        "schema": "preferred_certified_equivalence_set_nll",
        "preferred_action_loss_weight": 12.0,
        "preferred_action_uniform_loss_weight": 0.0,
        "preferred_action_tiebreak_loss_weight": 0.0,
        "preferred_action_rank_loss_weight": 0.0,
        "preferred_action_rank_margin": 1.0,
    }
