"""Install the pinned macOS OpenMP runtime into the active Python environment.

The F1 optional XGBoost and LightGBM wheels link against ``@rpath/libomp.dylib``.
This repository must not require a global Homebrew installation, so this tool
installs a verified Homebrew bottle into ``sys.prefix/lib`` and ad-hoc signs the
rewritten Mach-O binary. It is intentionally explicit and macOS/arm64-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.request import Request, urlopen


LIBOMP_VERSION = "22.1.8"
LIBOMP_BOTTLE_SHA256 = "7460e688895afb5df8c5f22a9e0ba2bffb0e46df265afe68eac56d538cd2496f"
GHCR_REPOSITORY = "homebrew/core/libomp"


def _download(url: str, *, headers: dict[str, str] | None = None) -> bytes:
    request = Request(url, headers=headers or {})
    with urlopen(request, timeout=60) as response:  # noqa: S310 - pinned host and digest
        return response.read()


def _anonymous_ghcr_token() -> str:
    payload = _download(
        "https://ghcr.io/token?service=ghcr.io&scope="
        f"repository:{GHCR_REPOSITORY}:pull"
    )
    token = str(json.loads(payload)["token"])
    if not token:
        raise RuntimeError("GHCR returned an empty anonymous pull token")
    return token


def install_openmp_runtime(*, prefix: Path | None = None, force: bool = False) -> Path:
    if sys.platform != "darwin" or platform.machine() != "arm64":
        raise RuntimeError("the pinned bootstrap currently supports macOS arm64 only")
    target_prefix = Path(prefix or sys.prefix).resolve()
    destination = target_prefix / "lib" / "libomp.dylib"
    if destination.exists() and not force:
        return destination

    token = _anonymous_ghcr_token()
    blob_url = f"https://ghcr.io/v2/{GHCR_REPOSITORY}/blobs/sha256:{LIBOMP_BOTTLE_SHA256}"
    payload = _download(
        blob_url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.oci.image.layer.v1.tar+gzip",
        },
    )
    actual_digest = hashlib.sha256(payload).hexdigest()
    if actual_digest != LIBOMP_BOTTLE_SHA256:
        raise RuntimeError(
            f"libomp bottle digest mismatch: expected {LIBOMP_BOTTLE_SHA256}, got {actual_digest}"
        )

    with tempfile.TemporaryDirectory(prefix="f1-libomp-") as temp_dir:
        archive = Path(temp_dir) / "libomp.tar.gz"
        archive.write_bytes(payload)
        with tarfile.open(archive, mode="r:gz") as bundle:
            members = bundle.getmembers()
            for member in members:
                resolved = (Path(temp_dir) / member.name).resolve()
                if not resolved.is_relative_to(Path(temp_dir).resolve()):
                    raise RuntimeError("unsafe path in libomp bottle")
            bundle.extractall(temp_dir, filter="data")
        source = Path(temp_dir) / "libomp" / LIBOMP_VERSION / "lib" / "libomp.dylib"
        if not source.is_file():
            raise RuntimeError(f"pinned bottle is missing {source.name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    subprocess.run(
        ["install_name_tool", "-id", "@rpath/libomp.dylib", str(destination)],
        check=True,
    )
    subprocess.run(["codesign", "--force", "--sign", "-", str(destination)], check=True)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=None, help="Python environment prefix")
    parser.add_argument("--force", action="store_true", help="replace an existing local runtime")
    args = parser.parse_args()
    installed = install_openmp_runtime(prefix=args.prefix, force=bool(args.force))
    print(installed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


# Suggested commit name: fix(f1-runtime): bootstrap a pinned local OpenMP runtime
