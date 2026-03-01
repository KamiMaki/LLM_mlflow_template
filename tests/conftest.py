"""共用 test fixtures。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.utils.config import init_config, reset_config


@pytest.fixture(autouse=True)
def _reset_config():
    """每個測試後重置 config 狀態。"""
    yield
    reset_config()


@pytest.fixture()
def default_config():
    """載入預設設定。"""
    return init_config()


@pytest.fixture()
def tmp_data_dir(tmp_path: Path) -> Path:
    """建立含測試資料的暫存目錄。"""
    # JSON
    json_file = tmp_path / "sample.json"
    json_file.write_text(
        json.dumps({"key": "value", "items": [1, 2, 3]}, ensure_ascii=False),
        encoding="utf-8",
    )

    # CSV
    csv_file = tmp_path / "sample.csv"
    csv_file.write_text("name,age\nAlice,30\nBob,25\n", encoding="utf-8")

    # TXT
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("Hello World\nLine 2", encoding="utf-8")

    # Markdown
    md_file = tmp_path / "sample.md"
    md_file.write_text("# Title\n\nContent here.", encoding="utf-8")

    return tmp_path


@pytest.fixture()
def test_cases_file(tmp_path: Path) -> Path:
    """建立測試用 test cases JSON。"""
    cases = [
        {
            "input": {"system_prompt": "Be helpful.", "user_prompt": "Say hello"},
            "expected": "hello",
            "metadata": {"category": "greeting"},
        },
        {
            "input": {"system_prompt": "Be helpful.", "user_prompt": "Say goodbye"},
            "expected": "goodbye",
            "metadata": {"category": "farewell"},
        },
    ]
    f = tmp_path / "test_cases.json"
    f.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    return f
