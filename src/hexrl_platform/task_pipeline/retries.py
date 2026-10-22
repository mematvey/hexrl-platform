from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

PERMANENT_REASON = "permanent_error"
EXHAUSTED_REASON = "max_deliveries_reached"
TRANSIENT_REASON = "transient_error"


@dataclass(frozen=True, slots=True)
class RetryDecision:
    attempt: int
    retry: bool
    reason: str


def attempt_number(headers: Mapping[str, Any], queue_name: str) -> int:
    # счётчик живёт в самом сообщении (x-death), а не в памяти воркера: она не переживёт рестарт
    deaths = headers.get("x-death")
    if not isinstance(deaths, Sequence) or isinstance(deaths, str | bytes):
        return 1
    for death in deaths:
        if isinstance(death, Mapping) and death.get("queue") == queue_name:
            return int(death.get("count", 0)) + 1
    return 1


def decide(
    *, headers: Mapping[str, Any], queue_name: str, max_deliveries: int, permanent: bool
) -> RetryDecision:
    # постоянная ошибка не станет другой от повтора; временной даём max_deliveries попыток
    attempt = attempt_number(headers, queue_name)
    if permanent:
        return RetryDecision(attempt=attempt, retry=False, reason=PERMANENT_REASON)
    if attempt >= max_deliveries:
        return RetryDecision(attempt=attempt, retry=False, reason=EXHAUSTED_REASON)
    return RetryDecision(attempt=attempt, retry=True, reason=TRANSIENT_REASON)
