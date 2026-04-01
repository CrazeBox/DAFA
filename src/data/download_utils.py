"""Download helpers with terminal progress bars."""

from __future__ import annotations

import shutil
import urllib.request
from pathlib import Path

from tqdm import tqdm


def download_url_with_progress(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> Path:
    """Download a URL to disk with a tqdm progress bar."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(url) as response:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total and total.isdigit() else None

        with tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=destination.name,
        ) as progress:
            with open(destination, "wb") as handle:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    handle.write(chunk)
                    progress.update(len(chunk))

    return destination


def copy_if_missing(source: Path, destination: Path) -> Path:
    """Copy a local file only when the destination does not exist."""
    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        shutil.copy2(source, destination)
    return destination
