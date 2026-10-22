import pytest

from hexrl_platform.task_pipeline.retries import ATTEMPT_HEADER, attempt_number, decide

QUEUE = "hexrl.inference.tasks"


def test_confirmed_republish_keeps_its_attempt_without_broker_death_headers():
    assert attempt_number({ATTEMPT_HEADER: 2}, QUEUE) == 2
    decision = decide(
        headers={ATTEMPT_HEADER: 3}, queue_name=QUEUE, max_deliveries=3, permanent=False
    )
    assert decision.attempt == 3
    assert not decision.retry


def test_legacy_rejected_message_keeps_its_broker_attempt_count():
    headers = {"x-death": [{"queue": QUEUE, "count": 2}]}
    assert attempt_number(headers, QUEUE) == 3


@pytest.mark.parametrize("value", [None, 0, -1, True, "invalid"])
def test_invalid_attempt_header_does_not_crash_processing(value):
    assert attempt_number({ATTEMPT_HEADER: value}, QUEUE) == 1


def test_invalid_legacy_header_does_not_create_an_infinite_crash_loop():
    assert attempt_number({"x-death": [{"queue": QUEUE, "count": "invalid"}]}, QUEUE) == 1
