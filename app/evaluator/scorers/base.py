"""BaseScorer — 所有 scorer 的抽象基底類別。"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseScorer(ABC):
    """Scorer 抽象基底。

    子類別需實作 score() 方法。
    """

    @property
    def name(self) -> str:
        """Scorer 名稱，預設為 class 名稱。"""
        return self.__class__.__name__

    @abstractmethod
    def score(self, output: str, expected: str) -> dict[str, float | str]:
        """評分。

        Args:
            output: 模型實際輸出。
            expected: 預期輸出或參考資料。

        Returns:
            dict 包含 "score" (float) 及 "reason" (str)。
        """
        ...
