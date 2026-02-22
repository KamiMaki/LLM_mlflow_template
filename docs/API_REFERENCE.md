# API 參考文件

本文件列出 LLM Framework 的所有公開 API。函式簽名保持英文，說明文字使用繁體中文。

---

## 設定管理（`llm_framework.config`）

### `load_config(env, config_dir)`

```python
def load_config(env: str | None = None, config_dir: str | Path = "config") -> FrameworkConfig
```

**用途**：載入指定環境的 YAML 設定檔，解析環境變數，並將結果設為全域單例。

**參數**：
| 參數 | 型別 | 說明 |
|------|------|------|
| `env` | `str \| None` | 環境名稱（dev/test/stg/prod）。若為 None，從 `LLM_ENV` 環境變數讀取，預設 `"dev"` |
| `config_dir` | `str \| Path` | 設定檔目錄路徑，預設 `"config"` |

**回傳**：`FrameworkConfig` 實例

**例外**：`ConfigError` — 設定檔不存在或必要欄位缺失時拋出

```python
from llm_framework.config import load_config

# 載入開發環境設定
config = load_config("dev")

# 從環境變數 LLM_ENV 自動判斷
config = load_config()
```

---

### `get_config()`

```python
def get_config() -> FrameworkConfig
```

**用途**：取得目前作用中的全域設定。必須先呼叫 `load_config()`。

**回傳**：`FrameworkConfig` 實例

**例外**：`ConfigError` — 尚未呼叫 `load_config()` 時拋出

---

### `load_config_from_dict(raw, env)`

```python
def load_config_from_dict(raw: dict, env: str = "custom") -> FrameworkConfig
```

**用途**：從字典載入設定，適合測試時使用。

```python
config = load_config_from_dict({
    "llm": {"url": "https://test.com/v1/chat/completions", "auth_token": "tok"},
    "mlflow": {"enabled": False}
}, env="test")
```

---

### `reset_config()`

```python
def reset_config() -> None
```

**用途**：重設全域設定單例。主要用於測試之間的清理。

---

### `FrameworkConfig`

```python
@dataclass(frozen=True)
class FrameworkConfig:
    llm: LLMConfig
    mlflow: MLflowConfig
    logging: LoggingConfig
    env: str
```

**用途**：頂層設定物件，不可變（frozen）。包含 LLM、MLflow、日誌三個子設定。

### `LLMConfig`

| 欄位 | 型別 | 預設值 | 說明 |
|------|------|--------|------|
| `url` | `str` | 必填 | LLM API 端點 URL |
| `auth_token` | `str` | 必填 | 認證 Token |
| `default_model` | `str` | `"gpt-4o"` | 預設模型 |
| `timeout` | `int` | `30` | 請求逾時秒數 |
| `max_retries` | `int` | `3` | 最大重試次數 |
| `temperature` | `float` | `0.7` | 預設溫度 |

### `MLflowConfig`

| 欄位 | 型別 | 預設值 | 說明 |
|------|------|--------|------|
| `tracking_uri` | `str` | `""` | MLflow 追蹤伺服器 URI |
| `experiment_name` | `str` | `"default"` | 預設實驗名稱 |
| `enabled` | `bool` | `True` | 是否啟用 MLflow |

---

## LLM 客戶端（`llm_framework.llm_client`）

### `LLMClient`

```python
class LLMClient:
    def __init__(self, config: FrameworkConfig | None = None)
```

**用途**：統一的 LLM 呼叫客戶端。自動讀取設定檔、處理認證、支援重試。

#### `chat()`

```python
def chat(
    self,
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs,
) -> LLMResponse
```

**用途**：發送聊天完成請求。

**參數**：
| 參數 | 說明 |
|------|------|
| `messages` | 訊息列表，每個元素包含 `role` 和 `content` |
| `model` | 覆蓋設定檔的預設模型 |
| `temperature` | 覆蓋設定檔的預設溫度 |
| `max_tokens` | 回應的最大 token 數 |

**回傳**：`LLMResponse`

**例外**：`LLMError` — 所有重試都失敗後拋出

```python
client = LLMClient()
response = client.chat([
    {"role": "system", "content": "你是一個有用的助手。"},
    {"role": "user", "content": "你好！"}
])
print(response.content)  # "你好！有什麼我可以幫助你的嗎？"
```

#### `achat()`

```python
async def achat(self, messages, model=None, temperature=None, max_tokens=None, **kwargs) -> LLMResponse
```

**用途**：`chat()` 的非同步版本，介面完全相同。

---

### `LLMResponse`

```python
@dataclass
class LLMResponse:
    content: str          # 回應文字
    model: str            # 使用的模型
    usage: TokenUsage     # Token 用量
    latency_ms: float     # 延遲（毫秒）
    raw_response: dict    # 原始 API 回應
```

### `TokenUsage`

```python
@dataclass
class TokenUsage:
    prompt_tokens: int       # 輸入 token 數
    completion_tokens: int   # 輸出 token 數
    total_tokens: int        # 總 token 數
```

---

## MLflow 日誌（`llm_framework.mlflow.logger`）

所有函式在 MLflow 停用或不可用時自動降級為 no-op（不做任何事），不會影響主流程。

### `log_llm_call()`

```python
def log_llm_call(prompt, response, model, token_usage, latency_ms, params=None) -> None
```

**用途**：記錄一次 LLM 呼叫的完整細節到當前 MLflow Run。

### `log_params(params)`

```python
def log_params(params: dict[str, Any]) -> None
```

**用途**：批量記錄參數（例如模型設定、超參數）。

### `log_metrics(metrics, step)`

```python
def log_metrics(metrics: dict[str, float], step: int | None = None) -> None
```

**用途**：批量記錄指標（例如準確率、延遲）。

### `log_artifact(local_path, artifact_path)`

```python
def log_artifact(local_path: str | Path, artifact_path: str | None = None) -> None
```

**用途**：將本地檔案記錄為 MLflow artifact。

### `log_dict_artifact(data, filename)`

```python
def log_dict_artifact(data: dict[str, Any], filename: str) -> None
```

**用途**：將字典序列化為 JSON 並記錄為 artifact。

---

## MLflow 追蹤（`llm_framework.mlflow.tracer`）

### `@trace_llm_call`

```python
def trace_llm_call(func: F) -> F
```

**用途**：裝飾器，將函式包裝為 LLM 類型的追蹤 span。自動記錄輸入、輸出、延遲。

```python
@trace_llm_call
def call_llm(prompt: str) -> str:
    return client.chat([{"role": "user", "content": prompt}]).content
```

### `@trace_node(node_name)`

```python
def trace_node(node_name: str) -> Callable[[F], F]
```

**用途**：裝飾器，將 workflow 節點函式包裝為 CHAIN 類型的追蹤 span。

```python
@trace_node("preprocess")
def preprocess(state: WorkflowState) -> dict:
    return {"results": {"cleaned": True}}
```

### `@trace_workflow(workflow_name)`

```python
def trace_workflow(workflow_name: str) -> Callable[[F], F]
```

**用途**：裝飾器，將整個 workflow 包裝為頂層追蹤 span。所有子 span 都會歸屬於此。

### `span(name, span_type)`

```python
@contextmanager
def span(name: str, span_type: str = "CHAIN") -> Generator
```

**用途**：Context manager，建立自訂追蹤 span。可用的 span 類型：`"LLM"`, `"CHAIN"`, `"TOOL"`, `"RETRIEVER"`。

```python
with span("vector_search", span_type="RETRIEVER"):
    results = search_vectors(query)
```

---

## MLflow 評估（`llm_framework.mlflow.evaluator`）

### `Evaluator`

```python
class Evaluator:
    def __init__(self, experiment_name: str | None = None)
```

**用途**：LLM 輸出品質評估器。支援內建指標與自訂評估函式。

#### `evaluate()`

```python
def evaluate(
    self,
    data: pd.DataFrame,
    metrics: list[str | Callable],
    model_output_col: str = "response",
    target_col: str = "expected",
) -> EvaluationResult
```

**用途**：對資料集執行評估。

**參數**：
| 參數 | 說明 |
|------|------|
| `data` | 包含模型輸出和預期答案的 DataFrame |
| `metrics` | 指標列表，可以是字串（內建指標名稱）或自訂函式 |
| `model_output_col` | 模型輸出欄位名稱 |
| `target_col` | 預期答案欄位名稱 |

**回傳**：`EvaluationResult`

#### `compare()`

```python
def compare(self, run_ids: list[str], metric_keys: list[str]) -> pd.DataFrame
```

**用途**：比較多個 MLflow Run 的指標。

### `EvaluationResult`

```python
@dataclass
class EvaluationResult:
    metrics: dict[str, float]     # 匯總指標
    per_row: pd.DataFrame         # 逐行評估結果
    run_id: str | None            # MLflow Run ID
```

---

## MLflow 實驗管理（`llm_framework.mlflow.experiment`）

### `ExperimentManager`

```python
class ExperimentManager:
    def __init__(self, config: Any | None = None)
```

#### `get_or_create_experiment(name)`

```python
def get_or_create_experiment(self, name: str | None = None) -> str | None
```

**用途**：取得或建立 MLflow 實驗。回傳實驗 ID。

#### `start_run(run_name, tags)`

```python
@contextmanager
def start_run(self, run_name: str | None = None, tags: dict | None = None)
```

**用途**：Context manager，開始一個 MLflow Run。

```python
manager = ExperimentManager()
with manager.start_run(run_name="baseline_v1", tags={"version": "1.0"}):
    log_params({"model": "gpt-4o"})
    log_metrics({"accuracy": 0.95})
```

#### `list_runs(experiment_name, filter_string)`

```python
def list_runs(self, experiment_name=None, filter_string=None) -> pd.DataFrame
```

**用途**：列出實驗中的所有 Run，支援篩選。

#### `get_best_run(metric, experiment_name, ascending)`

```python
def get_best_run(self, metric: str, experiment_name=None, ascending=False) -> dict | None
```

**用途**：取得指定指標最佳的 Run。

---

## Workflow 狀態（`llm_framework.workflow.state`）

### `BaseState`

```python
class BaseState(TypedDict):
    messages: Annotated[list, add_messages]  # 訊息列表（自動合併）
    metadata: dict[str, Any]                  # 自訂中繼資料
```

### `LLMState`

繼承 `BaseState`，新增：

| 欄位 | 型別 | 說明 |
|------|------|------|
| `llm_response` | `str` | LLM 回應文字 |
| `token_usage` | `dict[str, int]` | Token 用量 |
| `error` | `str \| None` | 錯誤訊息 |

### `WorkflowState`

繼承 `BaseState`，新增：

| 欄位 | 型別 | 說明 |
|------|------|------|
| `current_step` | `str` | 目前步驟名稱 |
| `results` | `dict[str, Any]` | 各節點的輸出結果 |
| `retry_count` | `int` | 重試次數 |
| `error` | `str \| None` | 錯誤訊息 |

### 工廠函式

```python
def create_base_state(messages=None, metadata=None) -> BaseState
def create_llm_state(messages=None, metadata=None, llm_response="", ...) -> LLMState
def create_workflow_state(messages=None, metadata=None, current_step="", ...) -> WorkflowState
```

**用途**：建立帶有合理預設值的狀態實例。

---

## Workflow 基礎（`llm_framework.workflow.base`）

### `BaseWorkflow`

```python
class BaseWorkflow:
    def __init__(self, name: str, state_schema: type = WorkflowState)
```

**用途**：LangGraph StateGraph 的高階包裝，提供鏈式 API。

**方法**（皆回傳 `self`，支援鏈式呼叫）：

| 方法 | 說明 |
|------|------|
| `add_node(name, func)` | 新增節點 |
| `add_edge(source, target)` | 新增直接邊 |
| `add_conditional_edge(source, condition, targets)` | 新增條件路由邊 |
| `set_entry(node_name)` | 設定入口節點 |
| `set_finish(node_name)` | 設定結束節點（連接到 END） |
| `compile()` | 編譯 workflow |
| `run(input_state, config)` | 同步執行（自動 MLflow 追蹤） |
| `arun(input_state, config)` | 非同步執行 |

```python
workflow = (
    BaseWorkflow("my_flow", WorkflowState)
    .add_node("step1", step1_func)
    .add_node("step2", step2_func)
    .set_entry("step1")
    .add_edge("step1", "step2")
    .set_finish("step2")
    .compile()
)
result = workflow.run(create_workflow_state())
```

---

## 工具集（`llm_framework.workflow.tools`）

### Parser（`tools.parser`）

```python
def extract_json(text: str) -> str
```
**用途**：從可能包含 Markdown 程式碼區塊或其他文字的內容中擷取 JSON 字串。

```python
def parse_json(text: str, fix_common_errors: bool = True) -> dict | list
```
**用途**：解析 LLM 輸出的 JSON。若 `fix_common_errors=True`，自動修復尾隨逗號、單引號、未加引號的鍵名等。

**例外**：`JSONParseError`

```python
def safe_parse_json(text: str, default: Any = None) -> dict | list | Any
```
**用途**：安全解析 JSON，失敗時回傳預設值而非拋出例外。

### Validator（`tools.validator`）

```python
def validate_output(data: dict, schema: type[BaseModel]) -> BaseModel
def validate_or_none(data: dict, schema: type[BaseModel]) -> BaseModel | None
def validate_list(data: list[dict], schema: type[BaseModel], skip_invalid=False) -> list[BaseModel]
def get_validation_errors(data: dict, schema: type[BaseModel]) -> list[str]
```

### Retry（`tools.retry`）

```python
@with_retry(max_retries=3, backoff_base=1.0, backoff_factor=2.0)
def risky_function(): ...

@with_async_retry(max_retries=5)
async def async_risky(): ...

def calculate_backoff(attempt: int, base=1.0, factor=2.0) -> float
```

### Structured Output（`tools.structured_output`）

```python
def get_structured_output(
    client: LLMClient,
    messages: list[dict],
    output_schema: type[BaseModel],
    max_retries: int = 3,
) -> BaseModel
```

**用途**：呼叫 LLM 並將回應解析為 Pydantic 模型。解析失敗時自動附上錯誤訊息重試。

### Prompt Template（`tools.prompt_template`）

```python
class PromptTemplate:
    def __init__(self, template: str)
    def render(self, **kwargs) -> str
    @classmethod
    def from_file(cls, path: str) -> "PromptTemplate"

def render_prompt(template_str: str, **kwargs) -> str
def create_chat_messages(system_prompt=None, user_prompt=None, assistant_prompt=None) -> list[dict]
def render_chat_messages(system_template=None, user_template=None, **kwargs) -> list[dict]
```
