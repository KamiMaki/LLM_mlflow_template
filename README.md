# LLM Project Template

通用 LLM 後端服務模板 — 預建 logging、evaluation、dataloader 等模組，工程師只需處理資料輸入與建立 workflow 即可開始工作。

## 專案結構

```
LLM_template/
├── pyproject.toml
├── Dockerfile
├── llm_config.yaml             # LLM 服務設定（多模型 + 多環境 + Token Exchange）
│
├── llm_service/                # LLM SDK（Config 管理 + Model Factory）
│   ├── config.py               # LLMConfig, SharedConfig, ModelConfig, ResolvedModelConfig
│   ├── factory.py              # get_langchain_llm(), get_adk_model(), build_multimodal_messages()
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
│   │   ├── state.py            # BaseState, WorkflowState（支援多模型 + 圖片）
│   │   ├── nodes.py            # create_call_llm_node(), create_set_model_node(), create_prompt_assembly_node()
│   │   └── build_workflow.py   # 預建 workflow（含 build_multimodel_chain）
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
# J1 Token（必須，用於 J1→J2 token exchange）
export LLM_AUTH_TOKEN="your-j1-token"

# 或為每個模型設定獨立的 J1 token
export LLM_AUTH_TOKEN_QWEN3="qwen3-j1-token"
export LLM_AUTH_TOKEN_QWEN3VL="qwen3vl-j1-token"

# 環境 zone（可選，預設 DEV）
export LLM_ZONE="DEV"

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
from llm_service import LLMConfig
from app.workflow import WorkflowState, create_call_llm_node, create_set_model_node
from langgraph.graph import END, StateGraph

# 載入多模型配置
cfg = LLMConfig.from_yaml()

# 建構 workflow
graph = StateGraph(WorkflowState)
graph.add_node("call_llm", create_call_llm_node(
    system_prompt="你是助手",
))
graph.set_entry_point("call_llm")
graph.add_edge("call_llm", END)
compiled = graph.compile()

# 執行（指定模型和環境）
result = compiled.invoke({
    "messages": [("user", "Hello")],
    "llm_config": cfg,
    "model_alias": "QWEN3",
})
```

## LLM Service 架構

`llm_service` 支援多模型 + 多環境配置，統一從 `llm_config.yaml` 讀取設定：

### 配置結構

```yaml
# llm_config.yaml
shared_config:
  default_zone: "DEV"
  auth_urls:
    DEV:  "https://auth-dev.internal.com/api/token"
    TEST: "https://auth-test.internal.com/api/token"
    STG:  "https://auth-stg.internal.com/api/token"
    PROD: "https://auth-prod.internal.com/api/token"
  token_field: "access_token"
  auth_method: "bearer"

model_configs:
  QWEN3:
    j1_token: ""  # 建議用 LLM_AUTH_TOKEN_QWEN3 環境變數
    model_name: "qwen3"
    api_endpoints:
      DEV:  "https://llm-dev.internal.com/v1"
      PROD: "https://llm-prod.internal.com/v1"
    max_tokens: 4096
    temperature: 0.7

  QWEN3VL:
    j1_token: ""
    model_name: "qwen3-vl"
    api_endpoints:
      DEV:  "https://llm-vl-dev.internal.com/v1"
      PROD: "https://llm-vl-prod.internal.com/v1"
    max_tokens: 4096
    temperature: 0.3
```

### 使用方式

```
llm_config.yaml
    ↓
LLMConfig.from_yaml()
    ↓
cfg.resolve("QWEN3", zone="PROD")  ← 自動 J1→J2 Token Exchange
    ↓
ResolvedModelConfig (api_base, api_key, model_name, hyperparams)
    ↓
├── get_langchain_llm(cfg, model_alias="QWEN3")  → ChatLiteLLM  → LangGraph
├── get_adk_model(cfg, model_alias="QWEN3")      → LiteLlm      → Google ADK
├── get_litellm_kwargs(cfg, model_alias="QWEN3")  → dict          → litellm.completion()
└── get_openai_client(cfg, model_alias="QWEN3")   → OpenAI        → OpenAI SDK
```

```python
from llm_service import LLMConfig, get_langchain_llm, get_litellm_kwargs

cfg = LLMConfig.from_yaml()

# LangGraph workflow（指定模型 + zone）
llm = get_langchain_llm(cfg, model_alias="QWEN3", zone="PROD")

# 直接呼叫 litellm
kwargs = get_litellm_kwargs(cfg, model_alias="QWEN3VL")
response = litellm.completion(**kwargs, messages=[...])
```

### Token Exchange（強制驗證）

所有模型呼叫都必須經過 J1→J2 token exchange。每個 zone 有獨立的 auth URL，
token 自動快取，過期前自動刷新。

```python
# J1 token 優先級：
# 1. LLM_AUTH_TOKEN_QWEN3 環境變數
# 2. LLM_AUTH_TOKEN 環境變數（fallback）
# 3. llm_config.yaml 中的 j1_token
```

### Base64 圖片支援

```python
from llm_service import build_multimodal_messages, get_litellm_kwargs
import litellm

# 建構含圖片的 messages
messages = build_multimodal_messages(
    user_text="描述這張圖表的趨勢",
    image_base64="base64_encoded_string...",
    system_prompt="你是視覺分析專家",
)

# 呼叫 VL 模型
kwargs = get_litellm_kwargs(cfg, model_alias="QWEN3VL")
response = litellm.completion(**kwargs, messages=messages)
```

## Workflow 架構

### WorkflowState

```python
from app.workflow import WorkflowState

# WorkflowState 包含：
# - messages: 訊息列表（MessagesState）
# - llm_config: LLMConfig 實例
# - prompt_template: 當前 prompt 模板
# - prompt_variables: prompt 變數
# - model_alias: 當前模型別名（"QWEN3", "QWEN3VL"）
# - zone: 當前環境
# - image_base64: base64 圖片
# - metadata: 額外資料
```

### 多模型 Workflow 範例

```python
from langgraph.graph import END, StateGraph
from app.workflow import (
    WorkflowState,
    create_call_llm_node,
    create_set_model_node,
    create_prompt_assembly_node,
)

graph = StateGraph(WorkflowState)

# 切換到 QWEN3 進行文字分析
graph.add_node("set_text_model", create_set_model_node("QWEN3"))

# 組裝 prompt（從 MLflow 載入）
graph.add_node("assemble", create_prompt_assembly_node(prompt_name="analysis"))

# 呼叫 LLM
graph.add_node("call_llm", create_call_llm_node(system_prompt="你是分析師"))

# 切換到 QWEN3VL 進行圖片分析
graph.add_node("set_vl_model", create_set_model_node("QWEN3VL"))
graph.add_node("call_vl", create_call_llm_node(
    system_prompt="分析圖片",
    support_images=True,
))

# 連接 edges
graph.set_entry_point("set_text_model")
graph.add_edge("set_text_model", "assemble")
graph.add_edge("assemble", "call_llm")
graph.add_edge("call_llm", "set_vl_model")
graph.add_edge("set_vl_model", "call_vl")
graph.add_edge("call_vl", END)

compiled = graph.compile()

# 執行
result = compiled.invoke({
    "messages": [],
    "llm_config": cfg,
    "model_alias": "QWEN3",
    "image_base64": "base64_image_data...",
    "prompt_variables": {"table": "| col1 | col2 |..."},
})
```

## API 認證

設定 `API_AUTH_TOKEN` 環境變數即可啟用 Bearer token 認證。未設定時所有 API 皆可公開存取。

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

## Evaluation

使用 MLflow GenAI evaluate 進行模型評估：

```python
from app.evaluator import run_evaluation
from app.evaluator.scorers import response_not_empty, contains_keywords

eval_data = [
    {
        "inputs": {"question": "台灣的首都是哪裡？"},
        "expectations": {"expected": "台北", "keywords": "台北,首都"},
    },
]

results = run_evaluation(
    predict_fn=my_app,
    eval_data=eval_data,
    scorers=[response_not_empty, contains_keywords],
)
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
| [06_complex_multimodel_workflow](examples/06_complex_multimodel_workflow.ipynb) | 多模型 Workflow + Markdown 表格 + 圖片 + 評估 |

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
