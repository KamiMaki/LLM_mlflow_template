# Migrate Workflow

將使用者現有的 LangGraph workflow 移植到本專案中，使其可以使用 LLMService 和 MLflow evaluation。

## 使用方式

```
/migrate <workflow 檔案路徑或目錄>
```

## 執行步驟

### 1. 分析來源 workflow

讀取使用者提供的 workflow 檔案（$ARGUMENTS），分析：

- **State 定義**：使用了哪些 state 欄位（MessagesState 或自定義）
- **Node 函式**：每個 node 做了什麼（LLM 呼叫、工具、前後處理）
- **Edge 結構**：包含哪些條件分支和路由邏輯
- **LLM 呼叫方式**：使用了什麼 LLM client（OpenAI、ChatLiteLLM、langchain ChatModel 等）
- **外部依賴**：需要的 tools、API、資料來源

### 2. 建立新的 workflow 模組

在 `app/workflow/` 目錄下建立新的 workflow 模組：

```
app/workflow/
├── __init__.py          # 匯出 workflow
└── <workflow_name>.py   # 移植後的 workflow
```

### 3. 轉換 LLM 呼叫

將所有 LLM 呼叫替換為 `LLMService`：

**替換前（常見模式）**：
```python
# OpenAI
from openai import OpenAI
client = OpenAI()
response = client.chat.completions.create(model="gpt-4", messages=[...])

# LangChain ChatModel
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4")
response = llm.invoke(messages)

# ChatLiteLLM
from langchain_litellm import ChatLiteLLM
llm = ChatLiteLLM(model="gpt-4")
```

**替換後**：
```python
from llm_service import LLMService

service = LLMService()  # 自動從 llm_config.yaml 載入設定

# 在 node 中使用
def my_node(state: dict) -> dict:
    response = service.call_llm(
        user_prompt=user_text,
        system_prompt="...",
    )
    return {"messages": [AIMessage(content=response.content)]}
```

### 4. 確保 State 相容

- 如果原始 workflow 使用 `MessagesState`，直接保留
- 如果使用自定義 State，保留所有欄位定義
- 確保 state 欄位使用 LangGraph 的 `Annotated` 和 reducer 語法（如有需要）

### 5. 建立 evaluation 支援

為移植後的 workflow 建立可評估的 predict_fn：

```python
from app.evaluator import make_workflow_predict_fn

predict_fn = make_workflow_predict_fn(
    compiled_graph,
    input_key="question",  # 根據 eval_data 的 inputs key 調整
)
```

如果 workflow 使用自訂 state，提供 state_builder 和 output_parser：

```python
predict_fn = make_workflow_predict_fn(
    compiled_graph,
    state_builder=lambda question, **kw: {
        "messages": [("user", question)],
        "custom_field": "value",
    },
    output_parser=lambda state: state["result"],
)
```

### 6. 更新 `app/workflow/__init__.py`

匯出新的 workflow：

```python
from app.workflow.<name> import compiled_graph as <name>_graph
```

### 7. 驗證

- 確認 workflow 可以正常 `graph.invoke()` 執行
- 確認 `make_workflow_predict_fn()` 包裝後可以搭配 `run_evaluation()` 使用
- 如果原始 workflow 有測試，將測試也移植到 `tests/` 目錄

## 注意事項

- **不要修改** `llm_service/` 目錄中的程式碼
- 所有 LLM 呼叫統一使用 `LLMService`，不直接呼叫 OpenAI / Anthropic / LiteLLM
- 如果原始 workflow 使用了 API key 環境變數，改為在 `llm_config.yaml` 中設定
- 保留原始 workflow 的業務邏輯和條件分支，只替換 LLM 呼叫層
