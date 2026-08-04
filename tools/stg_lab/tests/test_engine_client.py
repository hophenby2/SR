import json
import socket
import threading

from stg_lab.engine import EngineClient
from stg_lab.protocol import Action


def test_engine_client_request_response() -> None:
    client_socket, server_socket = socket.socketpair()

    def serve() -> None:
        reader = server_socket.makefile("rb")
        request = json.loads(reader.readline())
        assert request["command"] == "step"
        assert request["action"]["move_x"] == -1
        assert "controller_overlay_state" not in request
        response = {"id": request["id"], "ok": True, "observation": {"frame": 4}}
        server_socket.sendall((json.dumps(response) + "\n").encode())
        reader.close()
        server_socket.close()

    thread = threading.Thread(target=serve)
    thread.start()
    with EngineClient(client_socket) as client:
        response = client.step(Action(move_x=-1), repeat=3)
    thread.join()
    assert response["observation"]["frame"] == 4


def test_engine_client_sends_controller_overlay_state_when_provided() -> None:
    client_socket, server_socket = socket.socketpair()
    state = {
        "schema_version": 1,
        "revision": 7,
        "region_phase_radii": [7.0, None, 8.4],
    }

    def serve() -> None:
        reader = server_socket.makefile("rb")
        request = json.loads(reader.readline())
        assert request["controller_overlay_state"] == state
        server_socket.sendall((json.dumps({
            "id": request["id"],
            "ok": True,
            "observation": {"frame": 1},
        }) + "\n").encode())
        reader.close()
        server_socket.close()

    thread = threading.Thread(target=serve)
    thread.start()
    with EngineClient(client_socket) as client:
        client.step(
            Action(shoot=True),
            controller_overlay_state=state,
        )
    thread.join()


def test_engine_client_stage_reset_request() -> None:
    client_socket, server_socket = socket.socketpair()

    def serve() -> None:
        reader = server_socket.makefile("rb")
        request = json.loads(reader.readline())
        assert request == {
            "id": 1,
            "command": "reset_stage",
            "stage": "Stage 5@Lunatic",
            "seed": 7,
            "player": "reimu_player",
            "options": {"lifeleft": 3},
        }
        server_socket.sendall((json.dumps({
            "id": 1,
            "ok": True,
            "observation": {"episode_frame": 0},
        }) + "\n").encode())
        reader.close()
        server_socket.close()

    thread = threading.Thread(target=serve)
    thread.start()
    with EngineClient(client_socket) as client:
        response = client.reset_stage(
            "Stage 5@Lunatic",
            seed=7,
            options={"lifeleft": 3},
        )
    thread.join()
    assert response["observation"]["episode_frame"] == 0


def test_engine_client_campaign_reset_request_has_no_replay_surface() -> None:
    client_socket, server_socket = socket.socketpair()

    def serve() -> None:
        reader = server_socket.makefile("rb")
        request = json.loads(reader.readline())
        assert request == {
            "id": 1,
            "command": "reset_campaign",
            "difficulty": "Lunatic",
            "seed": 17,
            "player": "reimu_player",
            "options": {},
        }
        server_socket.sendall((json.dumps({
            "id": 1,
            "ok": True,
            "reset": {"episode_kind": "campaign"},
            "observation": {"episode_frame": 0},
        }) + "\n").encode())
        reader.close()
        server_socket.close()

    thread = threading.Thread(target=serve)
    thread.start()
    with EngineClient(client_socket) as client:
        response = client.reset_campaign("Lunatic", seed=17)
    thread.join()
    assert response["reset"]["episode_kind"] == "campaign"


def test_engine_client_campaign_reset_rejects_nonempty_options() -> None:
    client_socket, server_socket = socket.socketpair()
    try:
        with EngineClient(client_socket) as client:
            try:
                client.reset_campaign(
                    "Lunatic",
                    seed=17,
                    options={"lifeleft": 99},
                )
            except ValueError as error:
                assert str(error) == "campaign reset options must be empty"
            else:  # pragma: no cover - assertion aid
                raise AssertionError("campaign options were accepted")
    finally:
        server_socket.close()


def test_engine_client_replay_reset_request() -> None:
    client_socket, server_socket = socket.socketpair()

    def serve() -> None:
        reader = server_socket.makefile("rb")
        request = json.loads(reader.readline())
        assert request == {
            "id": 1,
            "command": "reset_replay",
            "path": "Z:\\replays\\slot2.rep",
        }
        server_socket.sendall((json.dumps({
            "id": 1,
            "ok": True,
            "reset": {"episode_kind": "replay", "frame_count": 10},
            "observation": {"episode_frame": 1},
        }) + "\n").encode())
        reader.close()
        server_socket.close()

    thread = threading.Thread(target=serve)
    thread.start()
    with EngineClient(client_socket) as client:
        response = client.reset_replay("Z:\\replays\\slot2.rep")
    thread.join()
    assert response["reset"]["episode_kind"] == "replay"


def test_engine_client_replay_reset_rejects_invalid_path() -> None:
    client_socket, server_socket = socket.socketpair()
    try:
        with EngineClient(client_socket) as client:
            for path in ("", "bad\x00path"):
                try:
                    client.reset_replay(path)
                except ValueError:
                    pass
                else:  # pragma: no cover - assertion aid
                    raise AssertionError("invalid replay path was accepted")
    finally:
        server_socket.close()


def test_engine_client_attack_reset_sends_replay_name() -> None:
    client_socket, server_socket = socket.socketpair()

    def serve() -> None:
        reader = server_socket.makefile("rb")
        request = json.loads(reader.readline())
        assert request == {
            "id": 1,
            "command": "reset",
            "scenario": "okuu:Lunatic",
            "attack": 3,
            "seed": 42,
            "player": "reimu_player",
            "options": {},
            "replay_name": "boss3-analysis",
        }
        server_socket.sendall((json.dumps({
            "id": 1,
            "ok": True,
            "reset": {"replay": {"name": "boss3-analysis"}},
            "observation": {"episode_frame": 1},
        }) + "\n").encode())
        reader.close()
        server_socket.close()

    thread = threading.Thread(target=serve)
    thread.start()
    with EngineClient(client_socket) as client:
        response = client.reset(
            "okuu:Lunatic",
            3,
            seed=42,
            replay_name="boss3-analysis",
        )
    thread.join()
    assert response["reset"]["replay"]["name"] == "boss3-analysis"


def test_engine_client_stage_reset_sends_replay_name() -> None:
    client_socket, server_socket = socket.socketpair()

    def serve() -> None:
        reader = server_socket.makefile("rb")
        request = json.loads(reader.readline())
        assert request == {
            "id": 1,
            "command": "reset_stage",
            "stage": "Stage 5@Lunatic",
            "seed": 43,
            "player": "reimu_player",
            "options": {},
            "replay_name": "stage5-analysis",
        }
        server_socket.sendall((json.dumps({
            "id": 1,
            "ok": True,
            "reset": {"replay": {"name": "stage5-analysis"}},
            "observation": {"episode_frame": 1},
        }) + "\n").encode())
        reader.close()
        server_socket.close()

    thread = threading.Thread(target=serve)
    thread.start()
    with EngineClient(client_socket) as client:
        response = client.reset_stage(
            "Stage 5@Lunatic",
            seed=43,
            replay_name="stage5-analysis",
        )
    thread.join()
    assert response["reset"]["replay"]["name"] == "stage5-analysis"


def test_engine_client_save_replay_request() -> None:
    client_socket, server_socket = socket.socketpair()

    def serve() -> None:
        reader = server_socket.makefile("rb")
        request = json.loads(reader.readline())
        assert request == {
            "id": 1,
            "command": "save_replay",
            "finish": False,
            "reason": "attack_complete",
        }
        server_socket.sendall((json.dumps({
            "id": 1,
            "ok": True,
            "replay": {"saved": True, "verified": True},
        }) + "\n").encode())
        reader.close()
        server_socket.close()

    thread = threading.Thread(target=serve)
    thread.start()
    with EngineClient(client_socket) as client:
        response = client.save_replay(
            finish=False,
            reason="attack_complete",
        )
    thread.join()
    assert response["replay"] == {"saved": True, "verified": True}
