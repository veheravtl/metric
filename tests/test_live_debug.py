"""Проверки локального моста живой диагностики."""

import json
from pathlib import Path

import numpy as np

from aerial_mapper.live_debug import LiveDebugSession


def test_live_debug_session_is_readable_while_open(tmp_path: Path) -> None:
    """События и состояние должны читаться до завершения приложения.

    Это главное свойство моста: Codex не ждёт экспорта архива и не просит
    закрыть GUI, а читает уже сброшенные строки работающей сессии.
    """

    session = LiveDebugSession(tmp_path / "live")
    session.record_event(
        "tracking_sample",
        {
            "points": np.asarray([[1.5, 2.5], [np.nan, np.inf]]),
            "accepted": np.asarray([True, False]),
        },
    )
    session.write_state({"frame_index": 17, "tracking_active": True})

    pointer = json.loads(
        (tmp_path / "live/latest_session.json").read_text(encoding="utf-8")
    )
    assert pointer["session_id"] == session.session_id
    assert Path(pointer["events_path"]) == session.events_path

    events = [
        json.loads(line)
        for line in session.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["event_type"] for event in events] == [
        "session_started",
        "tracking_sample",
    ]
    sample = events[-1]["payload"]
    assert sample["points"] == [[1.5, 2.5], [None, None]]
    assert sample["accepted"] == [True, False]

    state = json.loads(session.state_path.read_text(encoding="utf-8"))
    assert state["latest_event_sequence"] == 2
    assert state["state"]["frame_index"] == 17
    session.close()


def test_incident_keeps_independent_state_snapshot(tmp_path: Path) -> None:
    """Инцидент не должен зависеть от последующих изменений ``state.json``."""

    session = LiveDebugSession(tmp_path / "live")
    incident_directory = session.create_incident(
        "tracking_failed",
        {"frame_index": 8, "landmark_ids": [1, 2, 3]},
        payload={"reason": "недостаточно маяков"},
    )
    session.write_state({"frame_index": 9})

    incident = json.loads(
        (incident_directory / "incident.json").read_text(encoding="utf-8")
    )
    assert incident["reason"] == "tracking_failed"
    assert incident["state"]["frame_index"] == 8
    assert incident["payload"]["reason"] == "недостаточно маяков"
    assert (
        json.loads(session.state_path.read_text(encoding="utf-8"))["state"][
            "frame_index"
        ]
        == 9
    )

    session.close()
    session.close()
    event_types = [
        json.loads(line)["event_type"]
        for line in session.events_path.read_text(encoding="utf-8").splitlines()
    ]
    assert event_types.count("session_closed") == 1
