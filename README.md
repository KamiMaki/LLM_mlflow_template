# LLM Project Template

通用 LLM 後端服務模板 — 預建 logging、evaluation、dataloader 等模組，工程師只需處理資料輸入與建立 workflow 即可開始工作。

## 專案結構

```
LLM_template/
├── pyproject.toml
├── Dockerfile
├── llm_config.yaml             # LLM 服務設定（API、模型、Token Exchange）
│
├── llm_service/                # LLM SDK（Config 管理 + Model Factory）
│   ├── config.py               # LLMConfig + AuthConfig (Pydantic)
│   ├── factory.py              # get_langchain_llm(), get_adk_model(), get_litellm_kwargs(), get_openai_client()
│   ├── auth.py                 # TokenExchanger（J1→J2 token 交換）
│   ├── client.py               # LLMClient（deprecated，向後相容）
│   └── models.py               # LLMResponse, TokenUsage
│
├── app/                        # 主 package
│   ├── main.py                 # create_app() + uvicorn 入口
│   ├── api/                    # FastAPI 框架
│   │   ├── auth.py             # Bearer token 認證
│   │   ├── router.py           # /health, /ready
│   │   └── models.py           # Base request/response
│   ├── agents/                 # Google ADK Agent
│   │   └── base_agent.py       # create_agent(), run_agent_sync()
│   ├── workflow/               # LangGraph workflow（直接使用原生 API）
│   │   ├── state.py            # BaseState (MessagesState)
│   │   ├── nodes.py            # create_call_llm_node()（使用 ChatLiteLLM）
│   │   └── build_workflow.py   # 預建 workflow 範例
│   ├── dataloader/             # 資料載入模組
│   │   ├── base.py             # BaseLoader (ABC)
│   │   ├── local.py            # LocalFileLoader
│   │   └── models.py           # LoadedData, LoaderConfig
│   ├── evaluator/              # MLflow GenAI Evaluation
│   │   ├── runner.py           # run_evaluation(), run_trace_evaluation()
│   │   └── scorers.py          # @scorer + LLM Judge（使用 litellm）
│   ├── logger/                 # Loguru 日誌 + MLflow 初始化
│   │   └── setup.py            # setup_logging(), get_logger(), init_mlflow()
│   ├── prompts/                # Prompt 管理
│   │   ├── manager.py          # PromptManager (MLflow Registry + local fallback)
│   │   └── optimize.py         # optimize_prompt() (GEPA)
│   └── utils/                  # 設定管理
│       └── config.py           # YAML config 載入
│
├── config/
│   └── config.yaml             # Template 設定（API、logging、MLflow、dataloader）
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
# LLM 連線（也可在 llm_config.yaml 中設定）
export LLM_API_BASE="http://localhost:11434/v1"
export LLM_API_KEY="your-api-key"
export LLM_MODEL="gpt-4o"

# 可選：啟用 API 認證
export API_AUTH_TOKEN="your-api-auth-token"
```

### 3. 啟動服務

```bash
# 啟動
python -m app.main

# 使用 uvicorn (支援 hot reload)
uvicorn app.main:app --reload
```

### 4. 執行測試

```bash
uv run pytest -v
```

### 5. 建立你的第一個 Workflow

```python
from app.workflow import BaseState, create_call_llm_node
from langgraph.graph import END, StateGraph

# create_call_llm_node 自動從 llm_config.yaml 取得 ChatLiteLLM
call_llm = create_call_llm_node(system_prompt="你是助手")

graph = StateGraph(BaseState)
graph.add_node("call_llm", call_llm)
graph.set_entry_point("call_llm")
graph.add_edge("call_llm", END)
compiled = graph.compile()

result = compiled.invoke({"messages": [("user", "Hello")]})
```

## LLM Service 架構

`llm_service` 是 Config 管理 + Model Factory，統一從 `llm_config.yaml` 讀取設定，輸出各框架原生物件：

```
llm_config.yaml (or ENV vars)
        │
        ▼
    LLMConfig
        │
        ├── get_langchain_llm()  → ChatLiteLLM  → LangGraph
        ├── get_adk_model()      → LiteLlm      → Google ADK
        ├── get_litellm_kwargs() → dict          → litellm.completion()
        └── get_openai_client()  → OpenAI        → OpenAI SDK
```

```python
from llm_service import get_langchain_llm, get_adk_model, get_litellm_kwargs, get_openai_client

# LangGraph workflow
llm = get_langchain_llm()

# Google ADK agent
model = get_adk_model()

# 直接呼叫 litellm
kwargs = get_litellm_kwargs()
response = litellm.completion(**kwargs, messages=[...])

# OpenAI SDK
client = get_openai_client()
```

### Token Exchange（可選）

適用於內部 API 需要先用 J1 token 換取 J2 token 的場景。在 `llm_config.yaml` 設定 `auth` 區段即可自動處理：

```yaml
auth:
  auth_url: "https://auth.internal.com/api/token"
  auth_token: ""  # 或用 LLM_AUTH_TOKEN 環境變數
  token_field: "access_token"
  expires_field: "expires_in"
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

## Config 管理

使用單一 YAML 設定檔，支援 `${ENV_VAR}` 和 `${ENV_VAR:default}` 環境變數語法。

```yaml
# config/config.yaml
project_name: "my-llm-service"

api:
  host: "0.0.0.0"
  port: 8000
  auth_token: "${API_AUTH_TOKEN:}"

logging:
  level: "INFO"

mlflow:
  enabled: true
  experiment_name: "default"

dataloader:
  base_path: "./data"
```

> **Note:** `llm_service` 有自己獨立的 config（`llm_config.yaml`），不包含在此設定檔中。

### 程式中取得設定

```python
from app.utils.config import init_config, get_config

cfg = init_config()                    # 載入預設 config/config.yaml
cfg = init_config("path/to/my.yaml")   # 載入指定檔案

# dot-notation 和 dict 存取皆可
cfg.api.port       # 8000
cfg["api"]["port"]  # 8000
```

## Evaluation

使用 MLflow GenAI evaluate 進行模型評估：

```python
from app.evaluator import run_evaluation
from app.evaluator.scorers import response_not_empty, contains_keywords

# 準備評估資料
eval_data = [
    {
        "inputs": {"question": "台灣的首都是哪裡？"},
        "expectations": {"expected": "台北", "keywords": "台北,首都"},
    },
]

# 執行評估
results = run_evaluation(
    predict_fn=my_app,
    eval_data=eval_data,
    scorers=[response_not_empty, contains_keywords],
)
```

### LLM Judge（使用 litellm）

```python
from app.evaluator.scorers import create_quality_judge, create_tone_judge

quality_judge = create_quality_judge()
tone_judge = create_tone_judge()
```

## Prompt 管理

```python
from app.prompts import PromptManager

pm = PromptManager(cfg)

# 註冊 prompt (存到 MLflow 或 local)
pm.register("summarize", "請摘要以下內容：\n\n{{ text }}")

# 載入並格式化
rendered = pm.load_and_format("summarize", text="一段很長的文字...")

# 匯出為 markdown
pm.export_all("./prompts")
```

## Example Notebooks

| Notebook | 主題 |
|----------|------|
| [01_hello_world](examples/01_hello_world.ipynb) | 基本設定與 LLM 呼叫 + MLflow Autolog |
| [02_custom_workflow](examples/02_custom_workflow.ipynb) | LangGraph 多步驟 Workflow |
| [03_evaluation](examples/03_evaluation.ipynb) | MLflow GenAI 評估框架 |
| [04_prompt_management](examples/04_prompt_management.ipynb) | Prompt 註冊、版本管理與自動優化 |
| [05_adk_agent_phoenix](examples/05_adk_agent_phoenix.ipynb) | Google ADK Agent + Phoenix Tracing |

## 技術棧

- **FastAPI** — API 框架
- **LangGraph** — Workflow 引擎（直接使用原生 API）
- **LiteLLM** — 統一 LLM Provider 介面
- **MLflow 3.x** — Autolog Tracing、Prompt Registry、GenAI Evaluation
- **Loguru** — 日誌
- **Pydantic** — 資料驗證
- **Tenacity** — 重試控制
- **Google ADK** — Agent 框架（可選）

## 授權

MIT
