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
