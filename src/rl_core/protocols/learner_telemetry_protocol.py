"""``LearnerTelemetryProtocol`` — the counters ``DQNLearner`` reads and publishes."""

from typing import Protocol


class LearnerTelemetryProtocol(Protocol):
    """Transitions stored (read for the push-ratio throttle) and updates done (published)."""

    @property
    def transitions_pushed(self) -> int:
        """Transitions stored so far."""

    def note_update(self, update_count: int) -> None:
        """The learner's update count after each step."""
