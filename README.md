# LLM Project Template

通用 LLM 後端服務模板 — 預建 logging、evaluation、dataloader 等模組，工程師只需處理資料輸入與建立 workflow 即可開始工作。

## 專案結構

```
LLM_template/
├── pyproject.toml
├── Dockerfile
│
├── app/                        # 主 package
│   ├── main.py                 # create_app() + uvicorn 入口
│   ├── api/                    # FastAPI 框架
│   │   ├── auth.py             # Bearer token 認證
│   │   ├── router.py           # /health, /ready
│   │   └── models.py           # Base request/response
│   ├── workflow/               # LangGraph workflow
│   │   ├── base.py             # BaseWorkflow (builder pattern)
│   │   ├── state.py            # TypedDict state schemas
│   │   └── tools/              # parser, validator, structured_output
│   ├── dataloader/             # 資料載入模組
│   │   ├── base.py             # BaseLoader (ABC)
│   │   ├── local.py            # LocalFileLoader
│   │   └── models.py           # LoadedData, LoaderConfig
│   ├── evaluator/              # Evaluation 模組
│   │   ├── runner.py           # EvaluationRunner
│   │   ├── models.py           # TestCase, EvalResult
│   │   └── scorers/            # 內建 + 自定義 scorers
│   ├── logger/                 # Loguru 日誌
│   │   └── setup.py            # setup_logging(), get_logger()
│   ├── tracking/               # MLflow 整合
│   │   ├── setup.py            # init_mlflow()
│   │   ├── tracer.py           # trace decorators
│   │   ├── mlflow_logger.py    # MLflow logging
│   │   └── prompts.py          # PromptManager
│   └── utils/                  # 設定管理
│       └── config.py           # Hydra config 載入
│
├── llm_service/                # Mock LLM SDK（正式部署替換為真正 SDK）
│   ├── client.py               # LLMClient
│   └── models.py               # LLMResponse, TokenUsage
│
├── config/                     # Hydra 設定
│   ├── config.yaml             # 主設定
│   ├── llm/                    # LLM SDK 設定
│   ├── env/                    # 環境 overrides (dev/test/stg/prod)
│   └── ...                     # api, mlflow, logging, dataloader
│
├── tests/                      # 單元測試
├── examples/                   # Jupyter notebook 範例
├── prompts/                    # Prompt 模板 (部署用)
└── data/eval/                  # Evaluation test cases
```

## 快速開始

### 1. 安裝依賴

```bash
uv sync
```

### 2. 設定環境變數

```bash
export LLM_AUTH_TOKEN="your-api-key"
# 可選：啟用 API 認證
export API_AUTH_TOKEN="your-api-auth-token"
```

### 3. 啟動服務

```bash
# 開發環境 (預設)
python -m app.main

# 指定環境
python -m app.main env=prod

# 使用 uvicorn (支援 hot reload)
uvicorn app.main:app --reload
```

### 4. 執行測試

```bash
uv run pytest -v
```

### 5. 建立你的第一個 Workflow

```python
from app.workflow import BaseWorkflow, WorkflowState, create_workflow_state
from llm_service import LLMClient

client = LLMClient()

def my_node(state: dict) -> dict:
    response = client.chat("你是助手", state["metadata"]["user_input"])
    return {"results": {"output": response.content}}

workflow = (
    BaseWorkflow("my_workflow", WorkflowState)
    .add_node("process", my_node)
    .set_entry("process")
    .set_finish("process")
    .compile()
)

result = workflow.run(create_workflow_state(
    metadata={"user_input": "Hello"},
))
```

## API 認證

設定 `API_AUTH_TOKEN` 環境變數即可啟用 Bearer token 認證。未設定時所有 API 皆可公開存取。

```python
from fastapi import Depends
from app.api.auth import require_auth

@router.post("/chat")
async def chat(request: LLMRequest, token: str = Depends(require_auth)):
    ...
```

```bash
curl -H "Authorization: Bearer your-api-auth-token" http://localhost:8000/chat
```

## Config 管理 (Hydra)

使用 [Hydra](https://hydra.cc/) 進行設定管理，支援群組式設定組合與環境 override。

```yaml
# config/config.yaml
defaults:
  - llm: default
  - api: default
  - mlflow: default
  - logging: default
  - dataloader: default
  - env: dev          # 切換環境: dev/test/stg/prod
  - _self_
```

### 命令列 Override

```bash
python -m app.main llm.default_model=QWEN3 api.port=9000
```

### 程式中取得設定

```python
from app.utils.config import init_config, get_config, get_llm_config

cfg = init_config(overrides=["env=prod"])
llm_dict = get_llm_config()  # 給 llm_service SDK
```

## Evaluation

### 準備 Test Cases (`data/eval/my_cases.json`)

```json
[
  {
    "input": {
      "system_prompt": "你是一個摘要助手",
      "user_prompt": "請摘要以下文件..."
    },
    "expected": "關鍵字1,關鍵字2",
    "metadata": {"category": "summarization"}
  }
]
```

### 執行驗證

```python
from app.evaluator import EvaluationRunner
from app.evaluator.scorers import ContainsScorer, ExactMatchScorer

runner = EvaluationRunner()
results = runner.evaluate(
    workflow_fn=my_workflow_function,
    test_cases="data/eval/my_cases.json",
    scorers=[ContainsScorer(), ExactMatchScorer()],
)
print(f"平均分數: {results.metrics['avg_score']}")
print(f"通過率: {results.metrics['pass_rate']}")
# 結果自動記錄到 MLflow
```

### 自定義 Scorer

```python
from app.evaluator.scorers import BaseScorer

class MySimilarityScorer(BaseScorer):
    def score(self, output: str, expected: str) -> dict[str, float | str]:
        return {"score": 0.9, "reason": "High similarity"}
```

## Prompt 管理

```python
from app.tracking import PromptManager

pm = PromptManager(cfg)

# 註冊 prompt (DEV: 存到 MLflow)
pm.register("summarize", "請摘要以下內容：\n\n{{text}}")

# 載入並渲染
rendered = pm.render("summarize", text="一段很長的文字...")

# 部署前匯出為 markdown
pm.export_all("./prompts")
```

## Example Notebooks

| Notebook | 主題 |
|----------|------|
| [01_hello_world](examples/01_hello_world.ipynb) | 基本設定與 LLM 呼叫 |
| [02_custom_workflow](examples/02_custom_workflow.ipynb) | 建立多步驟 Workflow |
| [03_evaluation](examples/03_evaluation.ipynb) | 模型評估與計分 |
| [04_mlflow_tracking](examples/04_mlflow_tracking.ipynb) | MLflow 追蹤與 Prompt 管理 |

## 技術棧

- **FastAPI** — API 框架
- **Hydra** — 設定管理
- **LangGraph** — Workflow 引擎
- **MLflow** — Tracing、Prompt 管理、Evaluation 記錄
- **Loguru** — 日誌
- **Pydantic** — 資料驗證
- **Tenacity** — 重試控制
- **llm_service** — LLM 呼叫 SDK (外部依賴)

## 授權

MIT
