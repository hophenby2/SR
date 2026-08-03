from __future__ import annotations

import socket
from types import TracebackType
from typing import Any, Mapping, Self

from .protocol import Action, decode_message


class EngineProtocolError(RuntimeError):
    pass


class EngineClient:
    def __init__(self, connection: socket.socket, *, timeout: float = 30.0) -> None:
        self.connection = connection
        self.connection.settimeout(timeout)
        self._reader = connection.makefile("rb")
        self._next_id = 1

    @classmethod
    def connect(cls, host: str = "127.0.0.1", port: int = 24816, *, timeout: float = 30.0) -> Self:
        return cls(socket.create_connection((host, port), timeout=timeout), timeout=timeout)

    def close(self) -> None:
        self._reader.close()
        self.connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def request(self, command: str, **payload: Any) -> dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        message = {"id": request_id, "command": command, **payload}
        import json

        self.connection.sendall((json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8"))
        line = self._reader.readline()
        if not line:
            raise EngineProtocolError("engine bridge closed the connection")
        response = decode_message(line)
        if response["id"] != request_id:
            raise EngineProtocolError(
                f"response id {response['id']!r} does not match request {request_id}",
            )
        if response.get("ok") is False:
            raise EngineProtocolError(str(response.get("error", "unknown engine error")))
        return response

    def reset(
        self,
        scenario: str,
        attack: int,
        *,
        seed: int,
        player: str = "reimu_player",
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.request(
            "reset",
            scenario=scenario,
            attack=attack,
            seed=seed,
            player=player,
            options=dict(options or {}),
        )

    def reset_stage(
        self,
        stage: str,
        *,
        seed: int,
        player: str = "reimu_player",
        options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(stage, str) or not stage:
            raise ValueError("stage must be a nonempty string")
        return self.request(
            "reset_stage",
            stage=stage,
            seed=seed,
            player=player,
            options=dict(options or {}),
        )

    def ping(self) -> dict[str, Any]:
        return self.request("ping")

    def catalog(self) -> dict[str, Any]:
        return self.request("catalog")

    def observe(self) -> dict[str, Any]:
        return self.request("observe")

    def set_rendering(self, enabled: bool, *, every: int = 1) -> dict[str, Any]:
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a Boolean")
        if isinstance(every, bool) or not isinstance(every, int) or not 1 <= every <= 600:
            raise ValueError("every must be an integer in [1, 600]")
        return self.request("display", render=enabled, every=every)

    def step(self, action: Action, *, repeat: int = 1) -> dict[str, Any]:
        if repeat <= 0:
            raise ValueError("repeat must be positive")
        return self.request("step", action=action.to_dict(), repeat=repeat)
