from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

PERMANENT_REASON = "permanent_error"
EXHAUSTED_REASON = "max_deliveries_reached"
TRANSIENT_REASON = "transient_error"
ATTEMPT_HEADER = "x-attempt"


@dataclass(frozen=True, slots=True)
class RetryDecision:
    attempt: int
    retry: bool
    reason: str


def attempt_number(headers: Mapping[str, Any], queue_name: str) -> int:
    # Счётчик переносится при подтверждаемой публикации: новая публикация не обязана
    # сохранить брокерный x-death. Старые сообщения ещё могут приходить без x-attempt.
    attempt = headers.get(ATTEMPT_HEADER)
    if isinstance(attempt, int) and not isinstance(attempt, bool) and attempt >= 1:
        return attempt
    deaths = headers.get("x-death")
    if not isinstance(deaths, Sequence) or isinstance(deaths, str | bytes):
        return 1
    for death in deaths:
        if isinstance(death, Mapping) and death.get("queue") == queue_name:
            count = death.get("count", 0)
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                return count + 1
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
