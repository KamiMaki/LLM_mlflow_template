"""MLflow Prompt 管理 — 註冊、載入、匯出 prompt 模板。

DEV 環境: prompt 透過 MLflow 註冊管理（存在 mlruns/）。
部署環境: prompt 匯出為 local markdown，不依賴 MLflow。

Usage:
    from app.tracking.prompts import PromptManager

    pm = PromptManager(cfg)
    pm.register("summarize", "Summarize: {{text}}")
    template = pm.load("summarize")
    rendered = pm.render("summarize", text="Hello world")
    pm.export_all("./prompts")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Template

from app.logger import get_logger
from app.tracking.setup import is_mlflow_available

logger = get_logger(__name__)


class PromptManager:
    """管理 prompt 模板：MLflow 優先，local markdown fallback。"""

    def __init__(self, cfg=None) -> None:
        self._export_dir = Path("./prompts")
        if cfg is not None and hasattr(cfg, "mlflow"):
            export_dir = getattr(cfg.mlflow, "prompt_export_dir", "./prompts")
            self._export_dir = Path(export_dir)

    def register(self, name: str, template: str, commit_message: str = "") -> str | None:
        """註冊或更新 prompt 到 MLflow。

        Args:
            name: Prompt 名稱。
            template: Jinja2 模板字串。
            commit_message: 版本紀錄訊息。

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
                existing = mlflow.load_prompt(f"prompts:/{name}")
                if existing.template == template:
                    logger.debug(f"Prompt '{name}' unchanged, skipping update")
                    return existing.version
                prompt = mlflow.update_prompt(name=name, template=template, commit_message=commit_message)
                logger.info(f"Updated prompt '{name}' to version {prompt.version}")
                return prompt.version
            except Exception:
                prompt = mlflow.register_prompt(name=name, template=template, commit_message=commit_message)
                logger.info(f"Registered new prompt '{name}' version {prompt.version}")
                return prompt.version

        except Exception as e:
            logger.error(f"Failed to register prompt '{name}': {e}")
            self._save_local(name, template)
            return None

    def load(self, name: str, version: str | None = None) -> str:
        """載入 prompt 模板。MLflow 優先，local .md fallback。

        Args:
            name: Prompt 名稱。
            version: 指定版本，None 為最新版。

        Returns:
            模板字串。

        Raises:
            FileNotFoundError: 兩邊都找不到時。
        """
        if is_mlflow_available():
            try:
                import mlflow
                uri = f"prompts:/{name}" + (f"/{version}" if version else "")
                prompt = mlflow.load_prompt(uri)
                logger.debug(f"Loaded prompt '{name}' from MLflow")
                return prompt.template
            except Exception as e:
                logger.debug(f"MLflow load failed for '{name}': {e}, trying local")

        local_path = self._export_dir / f"{name}.md"
        if local_path.exists():
            logger.debug(f"Loaded prompt '{name}' from {local_path}")
            return local_path.read_text(encoding="utf-8")

        raise FileNotFoundError(f"Prompt '{name}' not found in MLflow or at {local_path}")

    def render(self, name: str, version: str | None = None, **variables: Any) -> str:
        """載入 prompt 並用 Jinja2 渲染。

        Args:
            name: Prompt 名稱。
            version: 指定版本。
            **variables: 模板變數。

        Returns:
            渲染後的字串。
        """
        template_str = self.load(name, version)
        return Template(template_str).render(**variables)

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
            client = mlflow.tracking.MlflowClient()
            for prompt_info in client.search_registered_prompts():
                name = prompt_info.name
                prompt = mlflow.load_prompt(f"prompts:/{name}")
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
