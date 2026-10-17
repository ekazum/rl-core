"""``ReplayTelemetryProtocol`` — the counters and rows the replay threads report to."""

from typing import Protocol


class ReplayTelemetryProtocol(Protocol):
    """Transition counter and buffer-stats rows."""

    @property
    def transitions_pushed(self) -> int:
        """Transitions stored so far."""

    def note_transitions(self, n: int) -> None:
        """Count ``n`` more stored transitions."""

    def log_buffer_stats(self, episode_index: int, metrics: dict) -> None:
        """Record one buffer-stats row."""
