"""Small JSON cache helpers for historical and forecast provider payloads."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class CacheRecord:
    path: Path
    payload: Mapping[str, Any]
    cache_hit: bool


class JsonHistoricalCache:
    """Deterministic file cache keyed by provider, endpoint, and query params."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    def key_path(self, provider: str, endpoint: str, params: Mapping[str, object]) -> Path:
        canonical = json.dumps(
            {"provider": provider, "endpoint": endpoint, "params": dict(params)},
            sort_keys=True,
            default=str,
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]
        return self.root / provider / f"{digest}.json"

    def read(self, path: Path, ttl_seconds: int | None = None) -> Mapping[str, Any] | None:
        if not path.exists():
            return None
        if ttl_seconds is not None:
            age_seconds = datetime.now().timestamp() - path.stat().st_mtime
            if age_seconds > ttl_seconds:
                return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                envelope = json.load(handle)
        except Exception:
            return None
        payload = envelope.get("payload") if isinstance(envelope, Mapping) else None
        return payload if isinstance(payload, Mapping) else None

    def write(self, path: Path, payload: Mapping[str, Any], metadata: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {"metadata": dict(metadata), "payload": payload}
        with path.open("w", encoding="utf-8") as handle:
            json.dump(envelope, handle, ensure_ascii=False, indent=2, default=str)
