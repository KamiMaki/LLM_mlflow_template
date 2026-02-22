# 快速開始指南

本指南將帶你從零開始，透過 5 個漸進式範例，一步步掌握 LLM Framework 的核心功能。

## 前置條件

開始之前，請確認你已經：

1. 安裝 Python 3.11 或更高版本
2. 安裝 uv 套件管理工具（`pip install uv`）
3. 設定環境變數 `LLM_AUTH_TOKEN`（你的 LLM API 認證 Token）

安裝框架：

```bash
uv sync --group dev
```

---

## 1. Hello World — 你的第一次 LLM 呼叫

**目標**：載入設定、建立 LLM 客戶端、發送第一個請求。

**前置條件**：
- 環境變數 `LLM_AUTH_TOKEN` 已設定
- 設定檔 `config/dev.yaml` 存在

**程式碼**：

```python
from llm_framework.config import load_config
from llm_framework.llm_client import LLMClient

# 載入開發環境設定
config = load_config("dev")
print(f"已載入環境: {config.env}")
print(f"LLM URL: {config.llm.url}")
print(f"預設模型: {config.llm.default_model}")

# 建立 LLM 客戶端
client = LLMClient()

# 發送聊天請求
messages = [
    {"role": "user", "content": "2 + 2 等於多少？"}
]

response = client.chat(messages)

print(f"\nLLM 回應: {response.content}")
print(f"使用模型: {response.model}")
print(f"Token 用量: {response.usage.total_tokens}")
print(f"延遲: {response.latency_ms:.2f}ms")
```

**預期輸出**：

```
已載入環境: dev
LLM URL: https://internal-llm-dev.company.com/v1/chat/completions
預設模型: gpt-4o

LLM 回應: 2 + 2 等於 4。
使用模型: gpt-4o
Token 用量: 18
延遲: 245.32ms
```

---

## 2. 測試單一節點 — 在 Workflow 中測試 Prompt

**目標**：建立一個 LangGraph 節點、搭配 MLflow 追蹤、獨立測試節點的效果。

**前置條件**：
- 已完成範例 1
- MLflow 伺服器運行中（或在設定中停用 `mlflow.enabled: false`）

**程式碼**：

```python
from llm_framework.config import load_config
from llm_framework.llm_client import LLMClient
from llm_framework.workflow.state import WorkflowState, create_workflow_state
from llm_framework.mlflow.tracer import trace_node
from llm_framework.mlflow.experiment import ExperimentManager

# 載入設定
load_config("dev")

# 初始化實驗管理器
manager = ExperimentManager()

# 定義一個 workflow 節點
@trace_node("greeting_generator")
def generate_greeting(state: WorkflowState) -> dict:
    """根據使用者名稱產生個人化問候語。"""
    client = LLMClient()

    # 從 metadata 取得使用者名稱
    user_name = state["metadata"].get("user_name", "訪客")

    # 呼叫 LLM
    messages = [
        {"role": "system", "content": "你是一個友善的助手。"},
        {"role": "user", "content": f"請為 {user_name} 產生一段溫暖的問候語。"}
    ]

    response = client.chat(messages, temperature=0.8)

    return {
        "results": {
            "greeting": response.content,
            "tokens_used": response.usage.total_tokens
        }
    }

# 獨立測試這個節點
with manager.start_run(run_name="test_greeting_node"):
    initial_state = create_workflow_state(
        metadata={"user_name": "Alice"}
    )

    result = generate_greeting(initial_state)

    print(f"問候語: {result['results']['greeting']}")
    print(f"Token 用量: {result['results']['tokens_used']}")
```

**預期輸出**：

```
問候語: 哈囉 Alice！很高興見到你，希望你今天過得愉快！
Token 用量: 25
```

你可以在 MLflow UI 中看到：
- 名為 "test_greeting_node" 的 Run
- "greeting_generator" 節點的追蹤 span
- Token 用量和延遲指標

---

## 3. 建立完整 Workflow — 多節點串接與條件路由

**目標**：建立多節點的 workflow，學會條件路由和 BaseWorkflow 的鏈式 API。

**前置條件**：已完成範例 1-2

**程式碼**：

```python
from llm_framework.config import load_config
from llm_framework.llm_client import LLMClient
from llm_framework.workflow.base import BaseWorkflow
from llm_framework.workflow.state import WorkflowState, create_workflow_state
from llm_framework.mlflow.experiment import ExperimentManager

load_config("dev")

# 定義各個節點

def classify_intent(state: WorkflowState) -> dict:
    """分類使用者查詢的意圖。"""
    client = LLMClient()
    user_query = state["metadata"]["query"]

    messages = [
        {"role": "system", "content": "將以下查詢分類為 'question' 或 'command'，只回傳一個詞。"},
        {"role": "user", "content": user_query}
    ]

    response = client.chat(messages, temperature=0.0)
    intent = response.content.strip().lower()

    return {
        "current_step": "classify",
        "results": {"intent": intent}
    }

def route_by_intent(state: WorkflowState) -> str:
    """根據意圖決定路由。"""
    intent = state["results"]["intent"]
    if "question" in intent:
        return "answer_question"
    return "execute_command"

def answer_question(state: WorkflowState) -> dict:
    """回答使用者的問題。"""
    client = LLMClient()
    query = state["metadata"]["query"]

    messages = [
        {"role": "system", "content": "你是一個有用的助手，請簡潔地回答問題。"},
        {"role": "user", "content": query}
    ]

    response = client.chat(messages)

    return {
        "current_step": "answer",
        "results": {**state["results"], "response": response.content}
    }

def execute_command(state: WorkflowState) -> dict:
    """執行使用者的指令。"""
    return {
        "current_step": "execute",
        "results": {**state["results"], "response": "指令執行功能尚未實作。"}
    }

# 組裝 Workflow
workflow = (
    BaseWorkflow("intent_router", WorkflowState)
    .add_node("classify", classify_intent)
    .add_node("answer_question", answer_question)
    .add_node("execute_command", execute_command)
    .set_entry("classify")
    .add_conditional_edge(
        "classify",
        route_by_intent,
        {"answer_question": "answer_question", "execute_command": "execute_command"}
    )
    .set_finish("answer_question")
    .set_finish("execute_command")
    .compile()
)

# 執行 Workflow
manager = ExperimentManager()

with manager.start_run(run_name="intent_routing_demo"):
    initial_state = create_workflow_state(
        metadata={"query": "法國的首都是哪裡？"}
    )

    final_state = workflow.run(initial_state)

    print(f"意圖: {final_state['results']['intent']}")
    print(f"回應: {final_state['results']['response']}")
```

**預期輸出**：

```
意圖: question
回應: 法國的首都是巴黎。
```

在 MLflow 中可以看到完整的 workflow 追蹤，包含 3 個 span（workflow、classify、answer_question）。

---

## 4. 評估 LLM 品質 — 自動化品質檢測

**目標**：準備測試資料、執行自動化評估、分析結果。

**前置條件**：已完成範例 1-3

**程式碼**：

```python
import pandas as pd
from llm_framework.config import load_config
from llm_framework.llm_client import LLMClient
from llm_framework.mlflow.evaluator import Evaluator

load_config("dev")

# 準備測試資料集
test_data = pd.DataFrame({
    "question": [
        "2 + 2 等於多少？",
        "法國的首都是哪裡？",
        "誰寫了《乘客和羅密歐》？",
    ],
    "expected": [
        "4",
        "巴黎",
        "莎士比亞",
    ]
})

# 用 LLM 產生回答
client = LLMClient()
responses = []

for question in test_data["question"]:
    messages = [{"role": "user", "content": question}]
    response = client.chat(messages, temperature=0.0)
    responses.append(response.content)

test_data["response"] = responses

# 建立評估器
evaluator = Evaluator(experiment_name="qa_evaluation")

# 自訂指標：檢查回答是否包含預期答案
def answer_contains_expected(row):
    expected = str(row["expected"]).lower()
    response = str(row["response"]).lower()
    return 1.0 if expected in response else 0.0

# 執行評估
result = evaluator.evaluate(
    data=test_data,
    metrics=["exact_match", "token_count", answer_contains_expected],
    model_output_col="response",
    target_col="expected"
)

# 顯示結果
print("=== 匯總指標 ===")
for metric, value in result.metrics.items():
    print(f"  {metric}: {value:.3f}")

print("\n=== 逐行結果 ===")
print(result.per_row[["question", "response", "expected"]])
```

---

## 5. 多環境部署 — 從開發到生產

**目標**：切換 dev/stg/prod 設定，了解各環境差異。

**前置條件**：已完成範例 1-4

**程式碼**：

```python
import os
from llm_framework.config import load_config

# 方法一：直接指定環境
print("=== 方法一：直接指定環境 ===")
config = load_config("dev")
print(f"環境: {config.env}")
print(f"LLM URL: {config.llm.url}")
print(f"日誌等級: {config.logging.level}")

# 方法二：使用環境變數
print("\n=== 方法二：使用環境變數 ===")
os.environ["LLM_ENV"] = "prod"
config = load_config()  # 從 LLM_ENV 讀取
print(f"環境: {config.env}")
print(f"LLM URL: {config.llm.url}")
print(f"日誌等級: {config.logging.level}")

# 方法三：執行時切換
print("\n=== 方法三：比較各環境設定 ===")
for env in ["dev", "stg", "prod"]:
    config = load_config(env)
    print(f"\n{env.upper()} 環境:")
    print(f"  URL: {config.llm.url}")
    print(f"  逾時: {config.llm.timeout}s")
    print(f"  溫度: {config.llm.temperature}")
    print(f"  MLflow 實驗: {config.mlflow.experiment_name}")
```

**生產部署檢查清單**：

1. 建立各環境設定檔（dev/stg/prod）
2. 設定環境變數（`LLM_AUTH_TOKEN`、`MLFLOW_TRACKING_URI`）
3. 先在 dev 環境充分測試
4. 用 `LLM_ENV=stg` 推上 staging 驗證
5. 用 `LLM_ENV=prod` 部署到生產
6. 在 MLflow UI 分別監控各環境的實驗

---

## 下一步

恭喜你完成了快速開始！接下來可以：

1. 閱讀 [API 參考文件](API_REFERENCE.md) 了解完整 API
2. 查看 [設定檔指南](CONFIGURATION.md) 了解進階設定
3. 探索 [範例 Notebook](../examples/) 了解更多使用場景
4. 閱讀 [常見問題](FAQ.md) 解決疑問
