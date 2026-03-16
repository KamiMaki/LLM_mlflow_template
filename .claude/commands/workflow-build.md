# Workflow Builder

根據使用者描述的 nodes 和連接方式，自動生成完整的 LangGraph workflow，包含驗證節點和錯誤處理。

## 使用方式

```
/workflow-build <workflow 描述>
```

範例：
```
/workflow-build 三個節點：parser 解析輸入 → checker 檢查格式 → summarizer 摘要輸出，checker 失敗時回到 parser
```

## 執行步驟

### 1. 解析使用者描述

從 `$ARGUMENTS` 中提取：

- **節點清單**：每個節點的名稱和職責
- **連接方式**：節點之間的 edge（包含條件分支）
- **State 需求**：根據節點功能推斷需要哪些 state 欄位

如果描述不夠明確，**主動詢問**：
- 每個節點的輸入/輸出是什麼？
- 條件分支的判斷邏輯是什麼？
- 是否需要特定的 LLM 模型？

### 2. 設計 State

根據節點需求設計 TypedDict state：

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class WorkflowState(TypedDict):
    # 必備欄位
    input_data: str                              # 原始輸入
    messages: Annotated[list, add_messages]       # LLM 對話紀錄（如需要）

    # 根據節點推斷的欄位
    parsed_result: dict                           # parser 輸出
    validation_errors: list[str]                  # checker 輸出
    final_output: str                             # 最終結果
```

### 3. 生成節點函式

每個節點遵循統一模式：

```python
from llm_service import LLMService

service = LLMService()

def node_name(state: WorkflowState) -> dict:
    """節點描述。"""
    # 從 state 取得輸入
    input_data = state["input_data"]

    # 執行邏輯（LLM 呼叫或純邏輯）
    response = service.call_llm(
        user_prompt=f"...\n{input_data}",
        system_prompt="...",
    )

    # 回傳更新的 state 欄位
    return {"parsed_result": response.content}
```

### 4. 自動加入驗證節點

對每個 LLM 呼叫節點，自動加入對應的驗證邏輯：

```python
def validate_node_output(state: WorkflowState) -> dict:
    """驗證前一個節點的輸出。"""
    errors = []
    result = state.get("parsed_result", "")

    if not result or not result.strip():
        errors.append("Output is empty")

    # 根據節點類型加入特定檢查
    # ...

    return {"validation_errors": errors}

def should_retry(state: WorkflowState) -> str:
    """條件路由：驗證通過則繼續，失敗則重試或報錯。"""
    errors = state.get("validation_errors", [])
    retry_count = state.get("retry_count", 0)

    if not errors:
        return "next_node"
    if retry_count < 3:
        return "retry"
    return "error"
```

### 5. 組裝 Graph

```python
from langgraph.graph import StateGraph, START, END

graph = StateGraph(WorkflowState)

# 加入所有節點
graph.add_node("parser", parser_node)
graph.add_node("validate_parser", validate_node_output)
graph.add_node("checker", checker_node)
graph.add_node("summarizer", summarizer_node)

# 加入 edges
graph.add_edge(START, "parser")
graph.add_edge("parser", "validate_parser")
graph.add_conditional_edges("validate_parser", should_retry, {
    "next_node": "checker",
    "retry": "parser",
    "error": END,
})
graph.add_edge("checker", "summarizer")
graph.add_edge("summarizer", END)

compiled_graph = graph.compile()
```

### 6. 生成檔案

在 `app/workflow/` 下建立：

```
app/workflow/
├── __init__.py              # 匯出 compiled_graph
├── <workflow_name>.py       # 完整 workflow
└── <workflow_name>_state.py # State 定義（如果 state 較複雜）
```

### 7. 生成評估支援

```python
from app.evaluator import make_workflow_predict_fn

predict_fn = make_workflow_predict_fn(
    compiled_graph,
    input_key="input_data",
    output_parser=lambda state: state["final_output"],
)
```

### 8. 生成測試

在 `tests/` 下建立基礎測試：

```python
def test_workflow_basic():
    result = compiled_graph.invoke({"input_data": "測試輸入"})
    assert result["final_output"]
    assert not result.get("validation_errors")
```

### 9. 驗證

- 確認 workflow 語法正確（import 無誤）
- 確認所有節點都有對應的 edge
- 確認沒有孤立節點或死循環
- 執行 `python -c "from app.workflow import ..."` 驗證 import

## 設計原則

- **每個 LLM 節點都配驗證**：不信任 LLM 的輸出，自動加入格式和內容檢查
- **重試機制**：驗證失敗時允許重試（預設 3 次），超過則走 error 路徑
- **State 最小化**：只定義節點真正需要的欄位
- **統一用 LLMService**：所有 LLM 呼叫走 `service.call_llm()`，不直接用其他 client
