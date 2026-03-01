"""LocalFileLoader — 從本地檔案系統載入資料。

支援: .json, .csv, .txt, .md

Usage:
    from app.dataloader.local import LocalFileLoader

    loader = LocalFileLoader()
    data = loader.load("documents/report.json")
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from app.dataloader.base import BaseLoader
from app.dataloader.models import LoadedData, LoaderConfig
from app.logger import get_logger

logger = get_logger(__name__)


class LocalFileLoader(BaseLoader):
    """從本地檔案系統載入資料。"""

    def load(self, source: str, **kwargs) -> LoadedData:
        """載入單一本地檔案。

        Args:
            source: 檔案路徑（相對於 base_path 或絕對路徑）。
        """
        path = self._resolve_path(source)
        suffix = path.suffix.lower()
        encoding = kwargs.get("encoding", self._config.encoding)

        if suffix == ".json":
            content = json.loads(path.read_text(encoding=encoding))
            content_type = "json"
        elif suffix == ".csv":
            with open(path, encoding=encoding, newline="") as f:
                content = list(csv.DictReader(f))
            content_type = "csv"
        elif suffix in (".md", ".markdown"):
            content = path.read_text(encoding=encoding)
            content_type = "markdown"
        else:
            content = path.read_text(encoding=encoding)
            content_type = "text"

        logger.debug(f"Loaded {content_type} file: {path}")
        return LoadedData(
            content=content,
            source=str(path),
            content_type=content_type,
            metadata={"size_bytes": path.stat().st_size},
        )

    def list_sources(self) -> list[str]:
        """列出 base_path 下所有檔案。"""
        base = Path(self._config.base_path)
        if not base.exists():
            return []
        return [str(p.relative_to(base)) for p in base.rglob("*") if p.is_file()]

    def _resolve_path(self, source: str) -> Path:
        """解析檔案路徑，支援相對/絕對路徑。"""
        p = Path(source)
        if not p.is_absolute():
            p = Path(self._config.base_path) / p
        if not p.exists():
            raise FileNotFoundError(f"Data source not found: {p}")
        return p
