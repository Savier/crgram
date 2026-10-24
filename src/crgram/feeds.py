"""Where the solver gets its board from: live recognition or a board file."""

from pathlib import Path
from typing import Protocol

from crgram import boardfile
from crgram.boardfile import BoardFile
from crgram.sources import Source, SourceError
from crgram.vision.recognize import CellRef, Recognizer, Scan


class Feed(Protocol):
    @property
    def name(self) -> str: ...

    def load(self) -> BoardFile:
        """Fetch the current board. Raises SourceError or BoardFormatError."""
        ...

    def changed(self) -> bool:
        """Whether a new load may give something new (polled by auto-refresh)."""
        ...

    def learn(self, ref: CellRef, letter: str) -> bool:
        """Remember a corrected letter for future recognition; False if not supported."""
        ...


class LiveFeed:
    """Screenshot from a source, recognized on every load."""

    def __init__(self, source: Source, recognizer: Recognizer) -> None:
        self.source = source
        self.recognizer = recognizer
        self._scan: Scan | None = None

    @property
    def name(self) -> str:
        return self.source.name

    def load(self) -> BoardFile:
        self._scan = self.recognizer.recognize(self.source.capture())
        return BoardFile(self._scan.board, source=self.source.name)

    def changed(self) -> bool:
        return True

    def learn(self, ref: CellRef, letter: str) -> bool:
        if self._scan is None or ref not in self._scan.glyphs:
            return False
        self.recognizer.learn_letter(self._scan, ref, letter)
        return True


class FileFeed:
    """A board file, reloaded when it changes on disk."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._mtime: float | None = None

    @property
    def name(self) -> str:
        return str(self.path)

    def _stat(self) -> float | None:
        try:
            return self.path.stat().st_mtime
        except OSError:
            return None

    def load(self) -> BoardFile:
        self._mtime = self._stat()
        try:
            return boardfile.load(self.path)
        except OSError as e:
            raise SourceError(f"cannot read {self.path}: {e}") from e

    def changed(self) -> bool:
        return self._stat() != self._mtime

    def mark_current(self) -> None:
        """Treat the file on disk as already loaded (after saving it ourselves)."""
        self._mtime = self._stat()

    def learn(self, ref: CellRef, letter: str) -> bool:
        return False
