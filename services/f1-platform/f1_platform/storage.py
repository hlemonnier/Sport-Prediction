"""Storage adapters for the F1 platform service."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from .schemas import F1Event


class JsonlEventStore:
    """Append-only raw event store for replay fixtures and local development."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def append(self, event: F1Event) -> Path:
        path = self.path_for_session(event.session_key)
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event.to_dict(), sort_keys=True))
                handle.write("\n")
        return path

    def replace(self, session_key: int | str, events: list[F1Event]) -> Path:
        path = self.path_for_session(session_key)
        with self._lock:
            with path.open("w", encoding="utf-8") as handle:
                for event in events:
                    handle.write(json.dumps(event.to_dict(), sort_keys=True))
                    handle.write("\n")
        return path

    def path_for_session(self, session_key: int | str) -> Path:
        safe_session = str(session_key).replace("/", "_")
        return self.root / f"{safe_session}.jsonl"
