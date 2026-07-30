from stg_lab.protocol import Action, Request, decode_message


def test_discrete_actions_round_trip() -> None:
    for index in range(18):
        assert Action.from_discrete(index).discrete == index


def test_request_round_trip() -> None:
    encoded = Request(7, "step", {"action": Action(move_x=1, slow=True).to_dict()}).encode()
    decoded = decode_message(encoded)
    assert decoded["id"] == 7
    assert decoded["command"] == "step"
    assert decoded["action"]["move_x"] == 1
