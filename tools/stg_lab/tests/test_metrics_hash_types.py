import numpy as np

from stg_lab.metrics import state_hash


def test_state_hash_preserves_boolean_type() -> None:
    assert state_hash({"value": False}) != state_hash({"value": 0})
    assert state_hash({"value": True}) != state_hash({"value": 1})
    assert state_hash({"value": np.bool_(False)}) == state_hash({"value": False})
