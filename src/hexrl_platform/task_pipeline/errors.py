from typing import Literal


class QueueOverloadedError(Exception):
    """Очередь полна или нет подтверждения: источник ждёт и повторяет, API отвечает 503"""

    def __init__(
        self, message: str = "", *, reason: Literal["nack", "timeout", "unavailable"] = "nack"
    ) -> None:
        super().__init__(message)
        self.reason = reason


class TopologyError(Exception):
    """Сообщение некуда маршрутизировать: ошибка конфигурации, повторять бессмысленно"""
