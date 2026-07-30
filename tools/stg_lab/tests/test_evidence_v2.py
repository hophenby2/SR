from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import pytest

from stg_lab.training import Demonstrations, load_checkpoint
from stg_lab.provenance import file_sha256


ROOT = Path(__file__).parents[1]
ARTIFACTS = ROOT / "artifacts"


def _load_experiment(name: str):
    path = ROOT / "experiments" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


agreement_v2 = _load_experiment("build_agreement_v2")
memory_v2 = _load_experiment("build_memory_benchmark_v2")
TRAIN_DATASET = agreement_v2.TRAIN_DATASET
HELDOUT_DATASET = agreement_v2.HELDOUT_DATASET
validate_bindings = agreement_v2.validate_bindings


def test_v2_agreement_inputs_are_bound_to_disjoint_external_split() -> None:
    train = Demonstrations.load(TRAIN_DATASET)
    heldout = Demonstrations.load(HELDOUT_DATASET)
    _model, metadata = load_checkpoint(ARTIFACTS / "policy_visible_v2.pt")

    binding = validate_bindings(metadata, train, heldout)

    assert binding["verified"]
    assert binding["heldout_seeds"] == {
        "stage5_boss3": [2001, 2002],
        "stage5_boss4": [2001, 2002],
    }
    assert binding["seed_overlap"] == {
        "stage5_boss3": [],
        "stage5_boss4": [],
    }


def test_v2_agreement_rejects_tampered_checkpoint_dataset_hash() -> None:
    train = Demonstrations.load(TRAIN_DATASET)
    heldout = Demonstrations.load(HELDOUT_DATASET)
    _model, metadata = load_checkpoint(ARTIFACTS / "policy_visible_v2.pt")
    tampered = deepcopy(metadata)
    tampered["training_data"]["sha256"] = "0" * 64

    with pytest.raises(ValueError, match="checkpoint training-data checksum"):
        validate_bindings(tampered, train, heldout)


def test_memory_benchmark_adapter_does_not_forward_authority_state() -> None:
    visible = object()

    class Controller:
        def __init__(self) -> None:
            self.received = None

        def select(self, value):
            self.received = value
            return "selected"

    controller = Controller()
    callback = memory_v2._visible_only_route_controller(controller)

    assert callback(object(), visible, object(), object()) == "selected"
    assert controller.received is visible


def test_memory_benchmark_loads_sqlite_artifact_without_modifying_it() -> None:
    before = file_sha256(memory_v2.DATABASE)
    artifact = memory_v2.load_route_library_artifact(memory_v2.LIBRARY_PATH)

    memories = memory_v2._load_route_memories(artifact.memory_ids)

    assert tuple(memory.id for memory in memories) == artifact.memory_ids
    assert file_sha256(memory_v2.DATABASE) == before
