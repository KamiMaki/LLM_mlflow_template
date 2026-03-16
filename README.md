# LLM Project Template

**開箱即用的 LLM 後端服務模板** — 預建所有基礎設施，工程師只需專注於資料輸入與 workflow 邏輯。

從第一行 `service.call_llm()` 開始，你的 LLM 呼叫就自動具備：token 交換認證、多環境切換、完整 tracing、評估框架。不需要自己處理 auth、不需要手動埋 log、不需要從零搭 evaluation pipeline。

---

## 為什麼使用這個模板？

### 零配置 LLM 呼叫

```python
from llm_service import LLMService

service = LLMService()
response = service.call_llm(
    user_prompt="請檢查這份合約",
    system_prompt="你是法務助手",
)
print(response.content)
```

一行初始化，自動完成：config 載入 → 模型解析 → J1→J2 token 交換 → LLM 呼叫 → 結構化回應。

### 完整支援公司內部 LLM 架構

`llm_service` 專為企業內部 LLM 環境設計：

- **J1→J2 Token Exchange**：自動處理公司內部的認證流程，token 快取 + 過期自動刷新
- **多環境 (DEV/TEST/STG/PROD)**：每個模型在每個環境有獨立的 endpoint 和 auth URL
- **多模型切換**：`service.set_model("QWEN3VL")` 一行切換，config 統一管理
- **統一 config 設定**：所有 LLM 相關設定集中在 `llm_config.yaml`，不散落在環境變數中

### 內建觀測性

每次 LLM 呼叫自動記錄到 MLflow：
- Token 用量、延遲、輸入/輸出完整追蹤
- LangGraph workflow 每個節點的執行軌跡
- 無需額外埋 code，autolog 自動生效

### 內建評估框架

不需要外部 LLM Judge API，用你自己的模型做評估：

```python
from app.evaluator import run_evaluation
from app.evaluator.scorers import response_not_empty, create_judge

results = run_evaluation(
    predict_fn=my_workflow,
    eval_data=test_cases,
    scorers=[response_not_empty, create_judge("correctness")],
)
```

---

## 適用場景

| 場景 | 本模板提供的支援 |
|------|-----------------|
| **文件處理 Pipeline** | Dataloader 模組 + LangGraph workflow，從 Word/Excel 到結構化輸出 |
| **多步驟 QA 檢查** | LangGraph 多節點 workflow，每個節點獨立呼叫不同模型 |
| **RAG 問答系統** | Dataloader 載入知識庫 + LLM 檢索增強 + 評估正確性 |
| **多模態分析** | QWEN3VL 支援圖片輸入，文字+圖片混合 prompt |
| **Agent 應用** | Google ADK 整合，tools + multi-turn 對話 |
| **Prompt 工程** | MLflow Prompt Registry 版本管理 + A/B 比較 |
| **模型評估 / 上線前驗證** | 內建 rule-based + LLM Judge scorer，MLflow 追蹤結果 |
| **批次資料處理** | Dataloader batch load + async LLM 呼叫 |

---

## 核心功能

### 1. 統一 LLM 呼叫 (`llm_service/`)

```python
service = LLMService()

# 基本呼叫
response = service.call_llm(user_prompt="...", system_prompt="...")

# 切換模型
service.set_model("QWEN3VL")
response = service.call_llm(user_prompt="描述圖片", image_base64=img_b64)

# Prompt template
response = service.call_llm(
    prompt_template="請檢查：\n{{ data }}",
    prompt_variables={"data": "..."},
)

# 覆寫超參數
response = service.call_llm(user_prompt="...", temperature=0.1, max_tokens=8192)

# Async
response = await service.acall_llm(user_prompt="...")
```

回應結構：
```python
response.content            # LLM 回應文字
response.model              # 使用的模型名稱
response.usage.total_tokens # Token 用量
response.latency_ms         # 延遲（毫秒）
response.reasoning_content  # 推理內容（如模型支援）
```

### 2. LangGraph Workflow (`app/workflow/`)

在 workflow 節點中使用 `LLMService`，自由組合多模型：

```python
from langgraph.graph import StateGraph, START, END
from llm_service import LLMService

service = LLMService()

def parse_node(state):
    response = service.call_llm(
        user_prompt=state["raw_text"],
        system_prompt="提取結構化資訊",
    )
    return {"parsed": response.content}

def check_node(state):
    service.set_model("QWEN3VL")
    response = service.call_llm(
        user_prompt=f"驗證：{state['parsed']}",
        image_base64=state.get("image"),
    )
    return {"check_result": response.content}

graph = StateGraph(dict)
graph.add_node("parse", parse_node)
graph.add_node("check", check_node)
graph.add_edge(START, "parse")
graph.add_edge("parse", "check")
graph.add_edge("check", END)
app = graph.compile()
```

### 3. Evaluation (`app/evaluator/`)

Rule-based 和 LLM Judge 混合使用：

```python
from app.evaluator.scorers import (
    response_not_empty,      # 非空檢查
    response_length_check,   # 長度合理性
    exact_match,             # 精確匹配
    contains_keywords,       # 關鍵字包含
    create_judge,            # LLM Judge（correctness, safety, relevance 等）
)
```

評估 workflow：
```python
from app.evaluator import run_evaluation, make_workflow_predict_fn

predict_fn = make_workflow_predict_fn(compiled_graph, input_key="question")
results = run_evaluation(predict_fn=predict_fn, eval_data=test_cases, scorers=scorers)
```

### 4. Data Loading (`app/dataloader/`)

可擴充的資料載入抽象：

```python
from app.dataloader import LocalFileLoader, LoaderConfig

loader = LocalFileLoader(LoaderConfig(base_path="./data"))
data = loader.load("report.json")       # 支援 .json, .csv, .txt, .md
all_data = loader.load_many(["a.json", "b.csv"])
```

自訂 loader 只需繼承 `BaseLoader`：
```python
class WordLoader(BaseLoader):
    def load(self, source, **kwargs) -> LoadedData: ...
    def list_sources(self) -> list[str]: ...
```

### 5. Prompt 管理 (`app/prompts/`)

```python
from app.prompts import PromptManager

pm = PromptManager(cfg)
pm.register("summarize", "請摘要：\n{{ text }}")              # MLflow Registry 或 local
rendered = pm.load_and_format("summarize", text="一段長文...")  # 載入 + 格式化
pm.export_all("./prompts")                                      # 匯出為 markdown
```

### 6. Google ADK Agent (`app/agents/`)

```python
from app.agents import create_agent, run_agent_sync

agent = create_agent(name="helper", instruction="回答用戶問題", service=service)
result = run_agent_sync(agent, "今天天氣如何？")
```

---

## Claude Code Skills

本專案內建三個 Claude Code Skills，加速新專案開發：

| Skill | 用途 |
|-------|------|
| `/workflow-build` | 描述 nodes + 連接方式 → 自動生成 LangGraph workflow（含驗證節點） |
| `/import-parser` | 從其他專案導入 parser/loader → 自動包裝為 BaseLoader 子類別 |
| `/project-init` | 新專案初始化：Word/Excel → parser → input state → workflow |
| `/migrate` | 將現有 workflow 移植到本模板，LLM 呼叫統一改用 LLMService |

```bash
# 範例：初始化一個合約審查系統
/project-init 合約審查系統，從 Word 合約和 Excel 條件表開始

# 導入現有的 parser
/import-parser ../my-project/parsers/word_parser.py

# 建立新 workflow
/workflow-build parser → validator → checker → summarizer
```

---

## 快速開始

### 1. 安裝

```bash
uv sync
```

### 2. 設定 LLM 連線

編輯 `llm_config.yaml`，填入你的模型資訊：

```yaml
default_model: "QWEN3"

shared_config:
  default_zone: "DEV"
  auth_urls:
    DEV:  "https://auth-dev.internal.com/api/token"
    PROD: "https://auth-prod.internal.com/api/token"

model_configs:
  QWEN3:
    j1_token: "your-j1-token"        # 或透過環境變數傳入
    model_name: "qwen3"
    api_endpoints:
      DEV:  "https://llm-dev.internal.com/v1"
      PROD: "https://llm-prod.internal.com/v1"
    max_tokens: 4096
    temperature: 0.7
```

所有 LLM 設定集中在這一個檔案中管理：模型名稱、endpoint、token、超參數。不需要記住多個環境變數。

### 3. 啟動服務

```bash
python -m app.main
# 或
uvicorn app.main:app --reload
```

### 4. 執行測試

```bash
uv run pytest -v
```

---

## 專案結構

```
LLM_template/
├── llm_config.yaml             # LLM 服務設定（模型、endpoint、認證）
├── config/config.yaml          # 應用程式設定（API、logging、MLflow）
├── pyproject.toml
│
├── llm_service/                # LLM 呼叫核心
│   ├── service.py              # LLMService — call_llm(), set_model()
│   ├── config.py               # 多模型 + 多環境 config 管理
│   ├── auth.py                 # J1→J2 Token Exchange（自動快取）
│   └── models.py               # LLMResponse, TokenUsage
│
├── app/
│   ├── main.py                 # FastAPI 入口
│   ├── api/                    # /health, /ready, Bearer auth
│   ├── workflow/               # LangGraph workflow（使用者建構）
│   ├── dataloader/             # 資料載入抽象（BaseLoader + LocalFileLoader）
│   ├── evaluator/              # MLflow GenAI Evaluation + LLM Judge
│   ├── prompts/                # Prompt Registry（MLflow + local fallback）
│   ├── agents/                 # Google ADK Agent 封裝
│   ├── logger/                 # Loguru + MLflow 初始化
│   └── utils/                  # YAML config 載入
│
├── tests/                      # 單元測試（95 tests）
├── examples/                   # Jupyter notebook 範例
└── data/                       # 資料目錄
```

## LLM Config 說明

`llm_config.yaml` 統一管理所有 LLM 連線設定：

| 區段 | 用途 |
|------|------|
| `shared_config.auth_urls` | 各環境的 J1→J2 認證端點 |
| `shared_config.default_zone` | 預設環境（DEV/TEST/STG/PROD） |
| `model_configs.*.j1_token` | 各模型的初始認證 token |
| `model_configs.*.api_endpoints` | 各模型在各環境的 API 位址 |
| `model_configs.*.temperature` | 模型超參數 |

**Token Exchange 流程**：`J1 token → POST auth_url → J2 token（自動快取，過期自動刷新）→ 呼叫 LLM API`

新增模型只需在 `model_configs` 中加一組設定，程式碼中用 `service.set_model("NEW_MODEL")` 切換。

## Example Notebooks

| Notebook | 主題 |
|----------|------|
| [01_hello_world](examples/01_hello_world.ipynb) | 基本 LLM 呼叫 + MLflow Autolog |
| [02_custom_workflow](examples/02_custom_workflow.ipynb) | LangGraph 多步驟 Workflow |
| [03_evaluation](examples/03_evaluation.ipynb) | MLflow GenAI 評估框架 |
| [04_prompt_management](examples/04_prompt_management.ipynb) | Prompt 版本管理 |
| [05_adk_agent_phoenix](examples/05_adk_agent_phoenix.ipynb) | Google ADK Agent + Phoenix Tracing |

## 技術棧

| 類別 | 技術 |
|------|------|
| API 框架 | FastAPI |
| Workflow 引擎 | LangGraph |
| LLM 介面 | LiteLLM |
| 觀測性 | MLflow 3.x（Autolog + Tracing + Evaluation） |
| 日誌 | Loguru |
| 資料驗證 | Pydantic |
| Agent 框架 | Google ADK（可選） |

## 授權

MIT
