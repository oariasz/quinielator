"""Descargas reproducibles con caché, hash y metadatos de procedencia."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class DownloadMetadata:
    """Trazabilidad mínima de un archivo externo."""

    name: str
    url: str
    downloaded_at: str
    sha256: str
    size_bytes: int
    license_note: str


class DataDownloader:
    """Descarga archivos atómicamente y conserva metadatos verificables."""

    def __init__(self, raw_directory: Path, timeout_seconds: int = 60) -> None:
        self.raw_directory = raw_directory
        self.timeout_seconds = timeout_seconds
        self.raw_directory.mkdir(parents=True, exist_ok=True)

    def download(
        self,
        *,
        name: str,
        url: str,
        filename: str,
        license_note: str,
        force: bool = False,
    ) -> Path:
        """Descarga ``url`` salvo que exista una copia local válida."""

        destination = self.raw_directory / filename
        if destination.exists() and not force:
            return destination

        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Quinielator/0.1 educational-project"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            content = response.read()

        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.write_bytes(content)
        temporary.replace(destination)

        metadata = DownloadMetadata(
            name=name,
            url=url,
            downloaded_at=datetime.now(UTC).isoformat(),
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            license_note=license_note,
        )
        metadata_path = destination.with_suffix(destination.suffix + ".metadata.json")
        metadata_path.write_text(
            json.dumps(asdict(metadata), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return destination
