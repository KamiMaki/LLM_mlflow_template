# LLM Framework

一套通用的 LLM 開發框架，整合 MLflow 追蹤、LangGraph 工作流程編排，支援多環境部署。

## 核心功能

- **統一的 LLM 客戶端**：自動重試、錯誤處理、Token 用量追蹤
- **多環境設定**：dev / test / stg / prod 無縫切換，透過 YAML 設定檔驅動
- **自動追蹤**：每次 LLM 呼叫自動記錄到 MLflow（輸入、輸出、延遲、Token 用量）
- **工作流程編排**：基於 LangGraph 的 workflow 模板，支援鏈式 API
- **結構化輸出**：搭配 Pydantic 驗證，自動重試直到取得正確格式
- **JSON 智慧解析**：自動修復 LLM 常見的 JSON 格式錯誤
- **Prompt 模板**：基於 Jinja2 的模板引擎，支援變數注入
- **品質評估**：內建評估工具，支援自訂指標
- **優雅降級**：MLflow 不可用時，框架照常運作不受影響

## 快速安裝

```bash
# 安裝 uv（如果還沒有的話）
pip install uv

# 安裝開發環境依賴
uv sync --group dev
```

## Hello World

```python
from llm_framework.config import load_config
from llm_framework.llm_client import LLMClient

# 載入設定
load_config("dev")

# 建立客戶端，發送請求
client = LLMClient()
response = client.chat([{"role": "user", "content": "你好！"}])
print(response.content)
```

## 文件

- [快速開始指南](docs/QUICKSTART.md) — 5 個漸進式範例，從零到部署
- [設定檔指南](docs/CONFIGURATION.md) — 環境設定與參數說明
- [API 參考文件](docs/API_REFERENCE.md) — 完整的 API 文件
- [常見問題](docs/FAQ.md) — 常見問題與解答

## 範例 Notebook

| 範例 | 說明 |
|------|------|
| [01_hello_world.ipynb](examples/01_hello_world.ipynb) | 基本 LLM 呼叫 |
| [02_single_node_test.ipynb](examples/02_single_node_test.ipynb) | 測試單一 Workflow 節點 |
| [03_full_workflow.ipynb](examples/03_full_workflow.ipynb) | 多節點 Workflow 與條件路由 |
| [04_evaluation.ipynb](examples/04_evaluation.ipynb) | LLM 品質評估 |
| [05_multi_env.ipynb](examples/05_multi_env.ipynb) | 多環境部署 |

## 專案結構

```
llm-framework/
├── pyproject.toml          # uv 專案設定
├── config/                 # 各環境設定檔
├── src/llm_framework/      # 框架原始碼
│   ├── config.py           # 設定管理
│   ├── llm_client.py       # LLM 客戶端
│   ├── mlflow/             # MLflow 整合模組
│   └── workflow/            # LangGraph 工作流程模組
├── tests/                  # 測試
├── docs/                   # 文件
└── examples/               # 範例 Notebook
```

## 授權

MIT
