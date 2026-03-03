"""MLflow Prompt 管理 — 使用 MLflow 3.x genai API 註冊、載入、格式化 prompt。

DEV 環境: prompt 透過 MLflow Prompt Registry 管理。
部署環境: prompt 匯出為 local markdown，不依賴 MLflow。

Usage:
    from app.prompts import PromptManager

    pm = PromptManager(cfg)
    pm.register("summarize", "Summarize: {{ text }}")
    formatted = pm.load_and_format("summarize", text="Hello world")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.logger import get_logger, is_mlflow_available

logger = get_logger(__name__)


class PromptManager:
    """管理 prompt 模板：MLflow Prompt Registry 優先，local markdown fallback。"""

    def __init__(self, cfg=None) -> None:
        self._export_dir = Path("./prompts")
        if cfg is not None and hasattr(cfg, "mlflow"):
            export_dir = getattr(cfg.mlflow, "prompt_export_dir", "./prompts")
            self._export_dir = Path(export_dir)

    def register(
        self,
        name: str,
        template: str,
        commit_message: str = "",
        model_config: dict[str, Any] | None = None,
    ) -> str | None:
        """註冊或更新 prompt 到 MLflow Prompt Registry。

        Args:
            name: Prompt 名稱。
            template: 模板字串（使用 {{ variable }} 語法）。
            commit_message: 版本紀錄訊息。
            model_config: 綁定的模型設定（model, temperature 等）。

        Returns:
            版本號字串，MLflow 不可用時回傳 None。
        """
        if not is_mlflow_available():
            logger.warning("MLflow unavailable, saving prompt locally only")
            self._save_local(name, template)
            return None

        try:
            import mlflow

            try:
                existing = mlflow.genai.load_prompt(f"prompts:/{name}")
                if existing.template == template:
                    logger.debug(f"Prompt '{name}' unchanged, skipping update")
                    return str(getattr(existing, "version", "latest"))

                kwargs = {"name": name, "template": template, "commit_message": commit_message}
                if model_config:
                    kwargs["model_config"] = model_config
                prompt = mlflow.genai.register_prompt(**kwargs)
                logger.info(f"Updated prompt '{name}' to version {prompt.version}")
                return str(prompt.version)
            except Exception:
                kwargs = {"name": name, "template": template, "commit_message": commit_message}
                if model_config:
                    kwargs["model_config"] = model_config
                prompt = mlflow.genai.register_prompt(**kwargs)
                logger.info(f"Registered new prompt '{name}' version {prompt.version}")
                return str(prompt.version)

        except Exception as e:
            logger.error(f"Failed to register prompt '{name}': {e}")
            self._save_local(name, template)
            return None

    def load(self, name: str, version: str | None = None) -> Any:
        """載入 prompt 物件。MLflow 優先，local fallback 回傳字串。

        Args:
            name: Prompt 名稱。
            version: 指定版本，None 為最新版。

        Returns:
            MLflow PromptVersion 物件（有 .format() 方法）或 template 字串。

        Raises:
            FileNotFoundError: 兩邊都找不到時。
        """
        if is_mlflow_available():
            try:
                import mlflow
                if version:
                    uri = f"prompts:/{name}/{version}"
                else:
                    uri = f"prompts:/{name}@latest"
                prompt = mlflow.genai.load_prompt(uri)
                logger.debug(f"Loaded prompt '{name}' from MLflow")
                return prompt
            except Exception as e:
                logger.debug(f"MLflow load failed for '{name}': {e}, trying local")

        local_path = self._export_dir / f"{name}.md"
        if local_path.exists():
            logger.debug(f"Loaded prompt '{name}' from {local_path}")
            return local_path.read_text(encoding="utf-8")

        raise FileNotFoundError(f"Prompt '{name}' not found in MLflow or at {local_path}")

    def load_and_format(self, name: str, version: str | None = None, **variables: Any) -> str:
        """載入 prompt 並格式化。

        Args:
            name: Prompt 名稱。
            version: 指定版本。
            **variables: 模板變數。

        Returns:
            格式化後的字串。
        """
        prompt = self.load(name, version)
        if not isinstance(prompt, str) and hasattr(prompt, "format"):
            return prompt.format(**variables)
        # local fallback: 簡易 {{ var }} 替換
        result = str(prompt)
        for key, value in variables.items():
            result = result.replace("{{ " + key + " }}", str(value))
        return result

    def export_all(self, output_dir: str | Path | None = None) -> list[str]:
        """匯出所有 MLflow 中的 prompt 為 local markdown。

        Args:
            output_dir: 匯出目錄，預設使用 config 中的 prompt_export_dir。

        Returns:
            匯出的檔案路徑列表。
        """
        if not is_mlflow_available():
            logger.warning("MLflow unavailable, cannot export prompts")
            return []

        try:
            import mlflow

            export_path = Path(output_dir) if output_dir else self._export_dir
            export_path.mkdir(parents=True, exist_ok=True)

            exported = []
            for prompt_info in mlflow.genai.search_prompts():
                name = prompt_info.name
                prompt = mlflow.genai.load_prompt(f"prompts:/{name}@latest")
                file_path = export_path / f"{name}.md"
                file_path.write_text(prompt.template, encoding="utf-8")
                exported.append(str(file_path))
                logger.info(f"Exported prompt '{name}' to {file_path}")

            return exported

        except Exception as e:
            logger.error(f"Failed to export prompts: {e}")
            return []

    def _save_local(self, name: str, template: str) -> None:
        """將 prompt 存為 local markdown。"""
        self._export_dir.mkdir(parents=True, exist_ok=True)
        path = self._export_dir / f"{name}.md"
        path.write_text(template, encoding="utf-8")
        logger.info(f"Saved prompt '{name}' locally to {path}")
