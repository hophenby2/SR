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
