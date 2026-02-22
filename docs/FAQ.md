# 常見問題（FAQ）

---

## 1. 如何安裝框架？

使用 `uv` 套件管理工具：

```bash
# 開發環境（包含完整 MLflow、LangGraph、測試工具）
uv sync --group dev

# 生產環境（僅安裝 mlflow-tracing + 必要依賴）
uv sync --group prod
```

如果尚未安裝 `uv`，請先執行：

```bash
pip install uv
```

---

## 2. 如何設定 LLM API 連線？

最少只需在設定檔中提供 `url` 和 `auth_token` 兩個欄位：

```yaml
# config/dev.yaml
llm:
  url: "https://your-llm-api.company.com/v1/chat/completions"
  auth_token: "${LLM_AUTH_TOKEN}"
```

然後設定環境變數：

```bash
export LLM_AUTH_TOKEN="your-token-here"
```

在 Python 中載入：

```python
from llm_framework.config import load_config
from llm_framework.llm_client import LLMClient

load_config("dev")
client = LLMClient()
response = client.chat([{"role": "user", "content": "你好"}])
```

---

## 3. MLflow 伺服器掛了怎麼辦？

**框架會自動降級處理**。所有 MLflow 相關操作都包裝在 try/except 中：

- 如果 MLflow 伺服器無法連線，所有 logging/tracing 函式會變成 no-op（不做任何事）
- 你的 LLM workflow 會繼續正常執行，不受影響
- 框架會在日誌中記錄警告訊息，方便事後排查

如果你想完全停用 MLflow，在設定檔中設定：

```yaml
mlflow:
  enabled: false
```

---

## 4. 如何切換不同的 LLM 模型？

有兩種方式：

**方式一：修改設定檔**

```yaml
llm:
  default_model: "gpt-4o-mini"  # 改為你想要的模型
```

**方式二：呼叫時覆蓋**

```python
# 單次呼叫使用不同模型
response = client.chat(
    messages=[{"role": "user", "content": "你好"}],
    model="gpt-4o-mini",       # 覆蓋設定檔的預設模型
    temperature=0.5             # 覆蓋設定檔的預設溫度
)
```

---

## 5. `mlflow` 和 `mlflow-tracing` 有什麼差別？

| 套件 | 大小 | 功能 | 適用環境 |
|------|------|------|----------|
| `mlflow` | 較大（含 UI、伺服器等） | 完整功能：logging、tracing、evaluation、UI | 開發 / 測試 |
| `mlflow-tracing` | 很小（輕量） | 僅追蹤功能：trace span、自動記錄 | 生產環境 |

在 `pyproject.toml` 中已經設定好：

- `dev` 群組安裝完整的 `mlflow`
- `prod` 群組只安裝 `mlflow-tracing`

這代表你的生產環境 Docker 映像會更小、啟動更快。

---

## 6. 如何新增自訂評估指標？

定義一個接受 DataFrame row 並回傳數值的函式：

```python
from llm_framework.mlflow.evaluator import Evaluator

# 自訂指標：檢查回答是否包含預期關鍵字
def contains_keyword(row):
    expected = str(row["expected"]).lower()
    response = str(row["response"]).lower()
    return 1.0 if expected in response else 0.0

# 自訂指標：回答長度是否合理（50-200 字元）
def reasonable_length(row):
    length = len(str(row["response"]))
    return 1.0 if 50 <= length <= 200 else 0.0

# 在評估中使用
evaluator = Evaluator()
result = evaluator.evaluate(
    data=test_data,
    metrics=["exact_match", contains_keyword, reasonable_length],
    model_output_col="response",
    target_col="expected"
)
```

---

## 7. 如何新增 Workflow 節點？

使用 `BaseWorkflow` 的鏈式 API：

```python
from llm_framework.workflow.base import BaseWorkflow
from llm_framework.workflow.state import WorkflowState

# 定義節點函式（接受 state，回傳更新的欄位）
def my_node(state: WorkflowState) -> dict:
    # 你的邏輯
    return {"results": {"output": "處理完成"}}

# 加入 workflow
workflow = (
    BaseWorkflow("my_workflow", WorkflowState)
    .add_node("my_node", my_node)
    .add_node("another_node", another_func)
    .set_entry("my_node")
    .add_edge("my_node", "another_node")
    .set_finish("another_node")
    .compile()
)
```

每個節點函式的規則：
- **輸入**：接收完整的 state 字典
- **輸出**：回傳一個字典，包含要更新的 state 欄位
- 不需要回傳完整 state，只需回傳改變的部分

---

## 8. 如何處理 LLM 回傳格式錯誤？

框架提供多層防護：

**第一層：JSON 自動修復**

```python
from llm_framework.workflow.tools.parser import parse_json

# 自動修復尾隨逗號、單引號等常見問題
data = parse_json('{"name": "test",}', fix_common_errors=True)
```

**第二層：結構化輸出（附自動重試）**

```python
from pydantic import BaseModel
from llm_framework.workflow.tools.structured_output import get_structured_output

class MyOutput(BaseModel):
    name: str
    score: float

# 自動重試最多 3 次，每次失敗會附上錯誤訊息回饋給 LLM
result = get_structured_output(client, messages, MyOutput, max_retries=3)
```

**第三層：安全解析（不拋例外）**

```python
from llm_framework.workflow.tools.parser import safe_parse_json

# 解析失敗時回傳預設值
data = safe_parse_json(llm_output, default={})
```

---

## 9. 如何部署到 Azure？

專案已包含 Dockerfile，支援多階段建置：

```bash
# 建置 Docker 映像
docker build -t llm-framework .

# 在本地測試
docker run -e LLM_AUTH_TOKEN="your-token" -e LLM_ENV=prod -p 8000:8000 llm-framework
```

部署到 Azure Container Apps：

```bash
# 登入 Azure
az login

# 建立容器應用
az containerapp up \
  --name my-llm-app \
  --source . \
  --env-vars LLM_AUTH_TOKEN=your-token LLM_ENV=prod
```

部署檢查清單：
1. 設定所有必要的環境變數（`LLM_AUTH_TOKEN`、`MLFLOW_TRACKING_URI`）
2. 確認使用 `prod` 依賴群組（較輕量）
3. 確認 `LLM_ENV=prod` 已設定
4. 測試健康檢查端點

---

## 10. 如何在測試中 mock LLM 呼叫？

使用 `unittest.mock` 來模擬 LLM 回應：

```python
from unittest.mock import patch, MagicMock
from llm_framework.llm_client import LLMResponse, TokenUsage

# 建立模擬回應
mock_response = LLMResponse(
    content="模擬的回應",
    model="gpt-4o",
    usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    latency_ms=100.0,
    raw_response={}
)

# 在測試中 mock LLMClient.chat
with patch.object(LLMClient, 'chat', return_value=mock_response):
    client = LLMClient()
    result = client.chat([{"role": "user", "content": "test"}])
    assert result.content == "模擬的回應"
```

或使用 `conftest.py` 中的 fixture（詳見 `tests/conftest.py`）。

---

## 11. 環境變數沒設定會怎樣？

如果設定檔中使用了 `${VAR_NAME}` 但對應的環境變數沒有設定，框架會在 `load_config()` 時立刻拋出 `ConfigError`：

```
ConfigError: Environment variable 'LLM_AUTH_TOKEN' is not set.
Please set it or update your config file.
```

這是**快速失敗**（fail-fast）策略：在應用程式啟動時就發現問題，而不是等到第一次 LLM 呼叫時才爆錯。

**解決方式**：

```bash
# Linux / macOS
export LLM_AUTH_TOKEN="your-token"

# Windows PowerShell
$env:LLM_AUTH_TOKEN = "your-token"

# Windows CMD
set LLM_AUTH_TOKEN=your-token
```

---

## 12. 如何查看 MLflow 的追蹤紀錄？

**方式一：MLflow UI**

如果你有 MLflow 追蹤伺服器：

```bash
# 啟動本地 MLflow UI
mlflow ui --port 5000
```

然後在瀏覽器開啟 `http://localhost:5000`，你可以看到：
- 所有實驗和 Run
- 每個 Run 的參數、指標、artifact
- Trace 檢視：完整的 span 樹狀結構，包含每個節點的輸入/輸出

**方式二：程式碼查詢**

```python
from llm_framework.mlflow.experiment import ExperimentManager

manager = ExperimentManager()
runs = manager.list_runs("my-experiment")
print(runs[["run_id", "status", "start_time"]])

# 取得最佳 Run
best = manager.get_best_run("accuracy")
print(f"最佳 Run: {best['run_id']}")
```

---

## 13. 如何建立 A/B 測試比較不同 Prompt？

使用 `ExperimentManager` 記錄不同 prompt 版本，再用 `Evaluator` 比較：

```python
from llm_framework.mlflow.experiment import ExperimentManager
from llm_framework.mlflow.evaluator import Evaluator

manager = ExperimentManager()

# Prompt A
with manager.start_run(run_name="prompt_v1", tags={"prompt_version": "v1"}):
    # 執行 prompt A，記錄結果
    log_params({"prompt": "請簡短回答以下問題："})
    # ... 執行評估 ...
    log_metrics({"accuracy": 0.85})

# Prompt B
with manager.start_run(run_name="prompt_v2", tags={"prompt_version": "v2"}):
    # 執行 prompt B，記錄結果
    log_params({"prompt": "你是專業的 QA 助手，請精確回答："})
    # ... 執行評估 ...
    log_metrics({"accuracy": 0.92})

# 比較結果
evaluator = Evaluator()
comparison = evaluator.compare(
    run_ids=["run_id_1", "run_id_2"],
    metric_keys=["accuracy", "avg_token_count"]
)
print(comparison)
```

---

## 14. 框架支援哪些 LLM 服務？

框架使用 **OpenAI 相容的 Chat Completion API**。只要你的 LLM 服務遵循以下格式，就可以使用：

- **端點**：`POST /v1/chat/completions`
- **請求格式**：`{"model": "xxx", "messages": [...], "temperature": 0.7}`
- **回應格式**：包含 `choices[0].message.content` 和 `usage` 欄位

相容的服務包括：
- OpenAI API
- Azure OpenAI
- 公司內部 LLM API（需符合上述格式）
- LiteLLM proxy
- vLLM
- 任何 OpenAI 相容的服務
