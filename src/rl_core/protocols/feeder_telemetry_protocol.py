"""``FeederTelemetryProtocol`` — the counters ``ActorFeeder`` reads to pace production."""

from typing import Protocol


class FeederTelemetryProtocol(Protocol):
    """Transitions stored and learner updates done."""

    @property
    def transitions_pushed(self) -> int:
        """Transitions stored so far."""

    @property
    def update_count(self) -> int:
        """Learner updates done so far."""
