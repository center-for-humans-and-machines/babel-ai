"""Live terminal progress for conversation turns."""

from __future__ import annotations

import shutil
import sys
from typing import Optional


class TurnProgress:
    """Render a turn counter, bar, and latest output preview."""

    def __init__(
        self,
        total: int,
        *,
        label: str = "Turn",
        enabled: Optional[bool] = None,
        stream=None,
    ) -> None:
        self.total = max(total, 1)
        self.label = label
        self.stream = stream or sys.stderr
        if enabled is None:
            enabled = hasattr(self.stream, "isatty") and self.stream.isatty()
        self.enabled = enabled
        self._started = False
        self._width = 80

    def update(self, current: int, speaker: str, content: str) -> None:
        """Refresh the progress bar and latest output preview."""
        if not self.enabled:
            return
        self._width = shutil.get_terminal_size((80, 20)).columns
        bar_line = self._bar_line(current)
        preview_line = self._preview_line(speaker, content)
        if self._started:
            self.stream.write("\033[2A\033[K")
        self.stream.write(f"{bar_line}\n{preview_line}\n")
        self.stream.flush()
        self._started = True

    def complete(self, message: str) -> None:
        """Clear the live display and print a final status line."""
        if not self.enabled:
            self.stream.write(f"{message}\n")
            self.stream.flush()
            return
        if self._started:
            self.stream.write("\033[2A\033[K")
            self._started = False
        self.stream.write(f"{message}\n")
        self.stream.flush()

    def _bar_line(self, current: int) -> str:
        """Build the progress bar line."""
        clamped = min(max(current, 0), self.total)
        ratio = clamped / self.total
        bar_width = 28
        filled = int(bar_width * ratio)
        bar = "#" * filled + "-" * (bar_width - filled)
        percent = int(ratio * 100)
        return f"{self.label} {clamped}/{self.total} " f"[{bar}] {percent:3d}%"

    def _preview_line(self, speaker: str, content: str) -> str:
        """Build a single-line preview of the latest output."""
        flattened = " ".join(content.split())
        prefix = f"{speaker}: "
        max_content = max(self._width - len(prefix) - 1, 12)
        if len(flattened) > max_content:
            flattened = flattened[: max_content - 3] + "..."
        return f"{prefix}{flattened}"
