"""抽象 BaseLoader — 所有自定義 loader 繼承此 class。

Usage:
    class MyKMSLoader(BaseLoader):
        def load(self, source: str, **kw) -> LoadedData:
            ...
        def list_sources(self) -> list[str]:
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from app.dataloader.models import LoadedData, LoaderConfig


class BaseLoader(ABC):
    """抽象資料載入器。

    Args:
        config: 載入器設定。若為 None 則使用預設值。
    """

    def __init__(self, config: LoaderConfig | None = None):
        self._config = config or LoaderConfig()

    @abstractmethod
    def load(self, source: str, **kwargs) -> LoadedData:
        """從指定來源載入資料。

        Args:
            source: 來源識別符（路徑、URL、ID 等）。

        Returns:
            LoadedData 包含內容與 metadata。
        """
        ...

    def load_many(self, sources: list[str], **kwargs) -> list[LoadedData]:
        """批次載入多個來源。子類別可 override 以優化效能。"""
        return [self.load(s, **kwargs) for s in sources]

    @abstractmethod
    def list_sources(self) -> list[str]:
        """列出可用的資料來源。"""
        ...
