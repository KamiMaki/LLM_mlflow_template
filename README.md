# LLM Project Template

**課內共用的 LLM 後端服務模板** — 讓同仁不再重複造輪子，專注在業務邏輯而非基礎建設。

## 這個專案能幫你什麼？

| 痛點 | 本模板的解決方案 |
|------|-----------------|
| 每個專案都要重寫 LLM 呼叫、token 認證、錯誤處理 | **一行 `service.call_llm()` 搞定**，認證 / retry / tracing 全自動 |
| 呼叫自研 AI Service（OCR、文件解析）要另外串接 | **`service.call_service()` 統一介面**，同樣享有認證與 retry |
| 上線前不確定 LLM 輸出品質，缺乏系統性評估方式 | **內建 Evaluation 框架**，rule-based + LLM Judge 混合評估 |
| 出了問題難以追蹤：哪個 prompt、哪個模型、花了多少 token | **MLflow Tracing 全自動紀錄**，敏感資料自動脫敏 |
| 想切換模型或環境，要改一堆設定 | **`llm_config.yaml` 統一管理**，一行切換模型 / 環境 |
| 每次新專案都從零開始搭 workflow | **LangGraph 整合 + Claude Code Skills**，快速生成 workflow 骨架 |

---

## 端到端使用流程

下圖說明從「拿到模板」到「上線驗證」的完整使用情境：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        LLM Project Template 使用流程                      │
└─────────────────────────────────────────────────────────────────────────┘

  ① 初始化專案                ② 設定 LLM 連線             ③ 開發業務邏輯
 ┌──────────────┐          ┌──────────────────┐        ┌──────────────────┐
 │  uv sync     │          │ 編輯               │        │ 撰寫 workflow    │
 │  安裝依賴     │ ──────▶  │ llm_config.yaml  │ ─────▶ │ 節點 / prompt    │
 │              │          │ 填入模型 & token   │        │ 串接 dataloader  │
 └──────────────┘          └──────────────────┘        └────────┬─────────┘
                                                                │
                    ┌───────────────────────────────────────────┘
                    ▼
  ④ 呼叫 LLM / AI Service    ⑤ 自動追蹤                  ⑥ 評估品質
 ┌──────────────────┐       ┌──────────────────┐        ┌──────────────────┐
 │ call_llm()       │       │ MLflow Tracing   │        │ run_evaluation() │
 │ call_service()   │ ────▶ │ 自動記錄：        │ ─────▶ │ rule-based 檢查  │
 │ (自動 retry +    │       │ prompt / token /  │        │ LLM Judge 評分   │
 │  token exchange) │       │ latency / 回應    │        │ 結果寫入 MLflow   │
 └──────────────────┘       └──────────────────┘        └────────┬─────────┘
                                                                 │
                    ┌────────────────────────────────────────────┘
                    ▼
  ⑦ 檢視結果 & 迭代           ⑧ 部署上線
 ┌──────────────────┐       ┌──────────────────┐
 │ MLflow UI 看報表  │       │ 切換 zone: PROD  │
 │ 比較不同 prompt   │ ────▶ │ uvicorn 啟動     │
 │ 調整模型 / 參數   │       │ FastAPI 服務上線  │
 └──────────────────┘       └──────────────────┘
```

### 步驟說明

| 步驟 | 你要做的事 | 模板幫你處理的事 |
|------|-----------|-----------------|
| ① 初始化 | `uv sync` 安裝依賴 | 所有套件版本已鎖定 |
| ② 設定連線 | 填寫 `llm_config.yaml`（模型名、token、endpoint） | Config 自動解析，支援環境變數 / 檔案掛載 token |
| ③ 開發邏輯 | 寫 LangGraph workflow 節點、設計 prompt | 提供 dataloader、prompt manager、workflow 範例 |
| ④ 呼叫 LLM | `service.call_llm()` / `service.call_service()` | J1→J2 認證、retry、timeout、錯誤處理全自動 |
| ⑤ 追蹤紀錄 | 不需要做任何事 | MLflow 自動記錄 prompt、回應、token、延遲，敏感資料脫敏 |
| ⑥ 評估品質 | 定義評估資料集 + 選擇 scorer | 執行評估、算分、結果寫入 MLflow |
| ⑦ 迭代優化 | 在 MLflow UI 比較不同版本 | 實驗追蹤、版本管理 |
| ⑧ 部署 | 切換 `default_zone: PROD` | endpoint / auth URL 自動切換 |

---

## 核心功能

### 1. 統一 LLM 呼叫 — `call_llm()`

```python
from llm_service import LLMService

service = LLMService()

# 基本呼叫
response = service.call_llm(user_prompt="請檢查這份合約", system_prompt="你是法務助手")
print(response.content)

# 切換模型
service.set_model("QWEN3VL")
response = service.call_llm(user_prompt="描述這張圖片", image_base64=img_b64)

# Prompt template
response = service.call_llm(
    prompt_template="請檢查：\n{{ data }}",
    prompt_variables={"data": contract_text},
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
response.reasoning_content  # 推理過程（如模型支援 reasoning）
```

### 2. 統一 AI Service 呼叫 — `call_service()`

除了 LLM，也能呼叫公司內部的自研 AI 服務（OCR、文件解析等），共享同一套認證與 retry 機制：

```python
# 在 llm_config.yaml 中設定 service_configs
response = service.call_service(
    service_name="IMAGE_EXTRACTION",
    payload={"image": base64_image, "type": "ocr"},
)
print(response.content)   # AI Service 回應
print(response.latency_ms)  # 延遲
```

`call_service()` 自動處理：
- J1→J2 Token Exchange（與 LLM 相同認證流程）
- Tenacity retry（指數退避）
- MLflow Tracing（自動記錄呼叫細節）
- 敏感資料脫敏（token、API key 不會出現在 trace 中）

### 3. 自動追蹤 — MLflow Tracing

每次呼叫 `call_llm()` / `call_service()` 自動記錄：

| 記錄項目 | 說明 |
|---------|------|
| Prompt 輸入 | system prompt + user prompt（完整保留） |
| LLM 回應 | content + reasoning_content |
| Token 用量 | prompt / completion / total tokens |
| 延遲 | 端到端毫秒數 |
| 模型資訊 | 模型名稱、環境、超參數 |
| 敏感資料 | **自動脫敏** — token、API key、密碼等欄位自動遮蔽 |

LangGraph workflow 中的每個節點也會被追蹤，形成完整的執行軌跡。

### 4. 評估框架 — Evaluation

內建 rule-based + LLM Judge 混合評估，**使用你自己的模型做 Judge，不需外部 API**：

```python
from app.evaluator import run_evaluation, make_workflow_predict_fn
from app.evaluator.scorers import (
    response_not_empty,      # 非空檢查
    response_length_check,   # 長度合理性
    exact_match,             # 精確匹配
    contains_keywords,       # 關鍵字包含
    create_judge,            # LLM Judge
)

# 評估 workflow
predict_fn = make_workflow_predict_fn(compiled_graph, input_key="question")
results = run_evaluation(
    predict_fn=predict_fn,
    eval_data=test_cases,
    scorers=[
        response_not_empty,
        create_judge("correctness"),
        create_judge("safety"),
    ],
)
```

可用的 LLM Judge 模板：`correctness`、`safety`、`relevance_to_query`、`answer_quality`、`professional_tone`，也可用 `create_llm_judge()` 自訂評估標準。

### 5. LangGraph Workflow

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

### 6. 其他模組

| 模組 | 說明 |
|------|------|
| **Data Loading** (`app/dataloader/`) | 可擴充的資料載入抽象，內建支援 .json / .csv / .txt / .md，繼承 `BaseLoader` 可自訂 |
| **Prompt 管理** (`app/prompts/`) | MLflow Registry 或 local 儲存，支援 `{{ var }}` 模板變數、版本管理、匯出 |
| **Google ADK Agent** (`app/agents/`) | 選配，支援 tool-use + multi-turn 對話 |

---

## 適用場景

| 場景 | 本模板提供的支援 |
|------|-----------------|
| **文件處理 Pipeline** | Dataloader 模組 + LangGraph workflow，從 Word/Excel 到結構化輸出 |
| **多步驟 QA 檢查** | LangGraph 多節點 workflow，每個節點獨立呼叫不同模型 |
| **RAG 問答系統** | Dataloader 載入知識庫 + LLM 檢索增強 + 評估正確性 |
| **多模態分析** | QWEN3VL 支援圖片輸入，文字+圖片混合 prompt |
| **自研 AI Service 整合** | `call_service()` 統一介面呼叫 OCR、文件解析等內部服務 |
| **Agent 應用** | Google ADK 整合，tools + multi-turn 對話 |
| **Prompt 工程** | MLflow Prompt Registry 版本管理 + A/B 比較 |
| **模型評估 / 上線前驗證** | 內建 rule-based + LLM Judge scorer，MLflow 追蹤結果 |

---

## 快速開始

### 1. 安裝

```bash
uv sync
```

### 2. 設定 LLM 連線

編輯 `llm_config.yaml`：

```yaml
default_model: "QWEN3"

shared_config:
  default_zone: "DEV"
  auth_urls:
    DEV:  "https://auth-dev.internal.com/api/token"
    PROD: "https://auth-prod.internal.com/api/token"
  retry:
    max_attempts: 3
    multiplier: 1
    min_wait: 2
    max_wait: 10

model_configs:
  QWEN3:
    j1_token: "your-j1-token"        # 或透過環境變數 / 檔案掛載
    model_name: "qwen3"
    api_endpoints:
      DEV:  "https://llm-dev.internal.com/v1"
      PROD: "https://llm-prod.internal.com/v1"
    max_tokens: 4096
    temperature: 0.7

service_configs:                       # 自研 AI Service（選配）
  IMAGE_EXTRACTION:
    j1_token: "your-service-token"
    api_endpoints:
      DEV:  "https://ocr-dev.internal.com/api/extract"
    timeout: 60
```

Token 來源優先順序：環境變數 `LLM_AUTH_TOKEN_{MODEL}` → config `j1_token` → 檔案路徑 `j1_token_path`

### 3. 啟動服務

```bash
uvicorn app.main:app --reload
```

### 4. 執行測試

```bash
uv run pytest -v
```

---

## Claude Code Skills

內建 Claude Code Skills，加速開發流程：

| Skill | 用途 |
|-------|------|
| `/workflow-build` | 描述 nodes + 連接方式 → 自動生成 LangGraph workflow |
| `/import-parser` | 從其他專案導入 parser/loader → 自動包裝為 BaseLoader 子類別 |
| `/project-init` | 新專案初始化：Word/Excel → parser → input state → workflow |
| `/migrate` | 將現有 workflow 移植到本模板，LLM 呼叫統一改用 LLMService |

```bash
# 範例：初始化一個合約審查系統
/project-init 合約審查系統，從 Word 合約和 Excel 條件表開始

# 建立新 workflow
/workflow-build parser → validator → checker → summarizer
```

---

## 專案結構

```
LLM_template/
├── llm_config.yaml             # LLM 服務設定（模型、endpoint、認證、retry）
├── config/config.yaml          # 應用程式設定（API、logging、MLflow）
├── pyproject.toml
│
├── llm_service/                # LLM 呼叫核心
│   ├── service.py              # LLMService — call_llm(), call_service(), set_model()
│   ├── config.py               # 多模型 + 多環境 + AI Service config 管理
│   ├── auth.py                 # J1→J2 Token Exchange（自動快取 + 過期刷新）
│   ├── models.py               # LLMResponse, AIServiceResponse, TokenUsage
│   └── trace.py                # MLflow Tracing + 敏感資料脫敏
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
├── tests/                      # 單元測試
├── examples/                   # Jupyter notebook 範例
└── docs/                       # 技術文件 + 使用指南
```

## LLM Config 說明

`llm_config.yaml` 統一管理所有連線設定：

| 區段 | 用途 |
|------|------|
| `shared_config.auth_urls` | 各環境的 J1→J2 認證端點 |
| `shared_config.default_zone` | 預設環境（DEV/TEST/STG/PROD） |
| `shared_config.retry` | retry 策略（次數、退避時間） |
| `shared_config.j1_token_path` | Pod 掛載 token 檔案路徑（適用 K8s） |
| `model_configs.*` | 各 LLM 模型的 token / endpoint / 超參數 |
| `service_configs.*` | 各 AI Service 的 token / endpoint / timeout |

**Token Exchange 流程**：`J1 token → POST auth_url → J2 token（自動快取，過期自動刷新）→ 呼叫 LLM / AI Service API`

新增模型：在 `model_configs` 中加一組設定，程式碼中 `service.set_model("NEW_MODEL")` 切換。
新增 AI Service：在 `service_configs` 中加一組設定，程式碼中 `service.call_service("SERVICE_NAME", payload)` 呼叫。

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
| LLM 介面 | LiteLLM（支援任意 LLM provider） |
| AI Service 呼叫 | httpx（async） |
| 觀測性 | MLflow 3.x（Tracing + Evaluation + 敏感資料脫敏） |
| Retry 機制 | Tenacity（指數退避） |
| 日誌 | Loguru |
| 資料驗證 | Pydantic |
| Agent 框架 | Google ADK（選配） |

## 授權

MIT
