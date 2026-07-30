from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class Action:
    move_x: int = 0
    move_y: int = 0
    slow: bool = False
    shoot: bool = True
    spell: bool = False

    def __post_init__(self) -> None:
        if self.move_x not in (-1, 0, 1) or self.move_y not in (-1, 0, 1):
            raise ValueError("movement components must be -1, 0, or 1")

    @property
    def discrete(self) -> int:
        direction = (self.move_y + 1) * 3 + self.move_x + 1
        return direction + (9 if self.slow else 0)

    @classmethod
    def from_discrete(cls, value: int, *, shoot: bool = True) -> "Action":
        if not 0 <= value < 18:
            raise ValueError("discrete action must be in [0, 18)")
        slow = value >= 9
        direction = value % 9
        return cls(
            move_x=direction % 3 - 1,
            move_y=direction // 3 - 1,
            slow=slow,
            shoot=shoot,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class Request:
    id: int
    command: str
    payload: Mapping[str, Any]

    def encode(self) -> bytes:
        data = {"id": self.id, "command": self.command, **self.payload}
        return (json.dumps(data, separators=(",", ":")) + "\n").encode("utf-8")


def decode_message(line: bytes | str) -> dict[str, Any]:
    if isinstance(line, bytes):
        line = line.decode("utf-8")
    message = json.loads(line)
    if not isinstance(message, dict):
        raise ValueError("protocol message must be a JSON object")
    if "id" not in message:
        raise ValueError("protocol message is missing id")
    return message
