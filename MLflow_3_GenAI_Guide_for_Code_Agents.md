# MLflow 3.x GenAI 完整指南（截至 3.10.0）

> **目標讀者**：Code Agent / LLM 專案開發者  
> **適用版本**：MLflow 3.0 – 3.10.0（2026-02-23 發佈）  
> **用途**：作為建立 LLM Template 專案的參考手冊，涵蓋 Tracing、Evaluation、Prompt Management、LangGraph Autolog 等核心 GenAI 功能

---

## 1. 安裝與環境設定

```bash
pip install mlflow==3.10.0
# 如需 Databricks 整合
pip install "mlflow[databricks]>=3.10"
# 生產環境輕量 tracing SDK（體積減少 95%）
pip install mlflow-tracing
```

啟動本地 Tracking Server：

```bash
# 基本啟動
mlflow server --host 0.0.0.0 --port 5000

# 3.10 新增：啟用多 workspace 支援
mlflow server --enable-workspaces
# 或設定環境變數
export MLFLOW_ENABLE_WORKSPACES=true
```

---

## 2. 核心架構：LoggedModel 與 Model-Centric 設計

MLflow 3 最根本的架構變革是引入 **LoggedModel** 實體，從過去以 Run 為中心轉為以 Model 為中心。LoggedModel 作為 metadata hub，將每個應用版本連結到對應的 Git commit、config、traces 與 evaluation runs。

```python
import mlflow

# 設定 active model，後續所有 traces 會自動連結
mlflow.set_active_model(name="my_chatbot_v2")

# 取得 model ID 以供追蹤
model_id = mlflow.get_active_model_id()
```

這個設計讓你能夠：
- 在不同版本間比較 traces 與 metrics
- 將 dev 環境與 production 環境的資料統一管理
- 建立完整的 lineage：model → runs → traces → prompts → evaluation metrics

---

## 3. Tracing（可觀測性）

### 3.1 Automatic Tracing（一行啟用）

MLflow 支援 **20+ GenAI 框架**的自動追蹤：

```python
import mlflow

# OpenAI
mlflow.openai.autolog()

# Anthropic
mlflow.anthropic.autolog()

# LangChain / LangGraph
mlflow.langchain.autolog()

# 其他支援的框架
mlflow.autogen.autolog()
mlflow.llamaindex.autolog()

# 全局啟用所有框架
mlflow.autolog()

# 停用
mlflow.langchain.autolog(disable=True)
```

### 3.2 Manual Tracing（精細控制）

```python
import mlflow

# 方式一：使用 decorator
@mlflow.trace
def my_llm_pipeline(question: str) -> str:
    # 你的邏輯
    return answer

# 方式二：使用 context manager 建立子 span
@mlflow.trace
def my_agent(query: str):
    with mlflow.start_span("retrieval") as span:
        span.set_inputs({"query": query})
        docs = retrieve(query)
        span.set_outputs({"docs": docs})
    
    with mlflow.start_span("generation") as span:
        span.set_inputs({"docs": docs, "query": query})
        answer = generate(docs, query)
        span.set_outputs({"answer": answer})
    
    return answer
```

### 3.3 LangGraph Autolog（Agent 追蹤）

LangGraph 的追蹤是透過 LangChain integration 實現的：

```python
from typing import Literal
import mlflow
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# 一行啟用 LangGraph tracing
mlflow.langchain.autolog()

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("LangGraph-Agent")

@tool
def get_weather(city: Literal["nyc", "sf"]):
    """取得天氣資訊"""
    if city == "nyc":
        return "It might be cloudy in nyc"
    elif city == "sf":
        return "It's always sunny in sf"

llm = ChatOpenAI(model="gpt-4o-mini")
graph = create_react_agent(llm, [get_weather])

# 每次 invoke 都會自動產生完整 trace
result = graph.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf?"}]}
)
```

**結合 Manual + Auto Tracing**：在 node 內加入子 span

```python
@mlflow.trace
def custom_node(state):
    with mlflow.start_span("custom_processing") as span:
        # 自定義邏輯
        span.set_attributes({"custom_key": "custom_value"})
    return state
```

### 3.4 Distributed Tracing（3.9+）

跨服務追蹤，透過 context propagation 維持 trace 連續性：

```python
from mlflow.tracing.distributed import get_trace_context, set_trace_context

# Service A：取得 context 並傳遞
context = get_trace_context()
# 透過 HTTP header 或 message queue 傳遞 context

# Service B：接收並設定 context
set_trace_context(context)
# 後續的 span 會自動連結到同一個 trace
```

### 3.5 Trace Cost Tracking（3.10 新增）

MLflow 會自動從 LLM span 提取模型資訊並計算成本，UI 中可直接查看每個 trace 的費用明細與整體花費趨勢。

### 3.6 Production Tracing

```python
# 啟用異步 logging，不影響應用效能
mlflow.config.enable_async_logging(True)

# 搜尋 traces
traces = mlflow.search_traces(
    filter_string="tags.model = 'gpt-4o-mini'",
    # 可加入時間範圍、狀態等過濾條件
)
```

---

## 4. Prompt Registry（Prompt 管理）

### 4.1 註冊與版本控制

```python
import mlflow

# 註冊 prompt（使用 {{ }} 作為變數語法）
prompt = mlflow.genai.register_prompt(
    name="customer_support_prompt",
    template="""\
You are a helpful customer support assistant.
Answer the following question: {{ question }}
Context: {{ context }}
""",
    commit_message="Initial version of customer support prompt",
)

print(f"Prompt: {prompt.name}, Version: {prompt.version}")
```

### 4.2 載入與使用

```python
# 載入最新版本
prompt = mlflow.genai.load_prompt("prompts:/customer_support_prompt@latest")

# 載入特定版本
prompt_v2 = mlflow.genai.load_prompt("prompts:/customer_support_prompt/2")

# 格式化
formatted = prompt.format(
    question="How do I reset my password?",
    context="User is on the settings page."
)
```

### 4.3 搜尋 Prompts（3.0.1+）

```python
prompts = mlflow.genai.search_prompts(filter_string="name LIKE '%support%'")
```

### 4.4 Prompt Model Configuration（3.8+）

Prompt 現在可以包含模型設定，讓 prompt 與模型參數綁定：

```python
prompt = mlflow.genai.register_prompt(
    name="qa_prompt",
    template="Answer: {{ question }}",
    model_config={
        "model": "gpt-4o-mini",
        "temperature": 0.1,
        "max_tokens": 2000,
    },
    commit_message="QA prompt with model config",
)
```

### 4.5 Prompt Optimization（3.5+ beta）

自動優化 prompt，使用 GEPA 演算法（基於 DSPy 的 MIPROv2）：

```python
from mlflow.genai.optimize import GepaPromptOptimizer, LLMParams
from mlflow.genai.scorers import scorer

@scorer
def exact_match(outputs, expectations):
    return 1.0 if outputs.strip() == expectations["answer"].strip() else 0.0

# 準備訓練與評估資料
train_data = [
    {"inputs": {"question": "What is 2+2?"}, "expectations": {"answer": "4"}},
    # ... 更多資料
]

# 執行優化（會自動註冊新版本到 Prompt Registry）
result = mlflow.genai.optimize_prompts(
    predict_fn=predict_fn,
    train_data=train_data,
    prompt_uris=[prompt.uri],
    optimizer=GepaPromptOptimizer(reflection_model="openai:/gpt-4o"),
    scorers=[exact_match],
)

print(result.prompt.uri)  # 優化後的 prompt URI
```

---

## 5. Evaluation（評估框架）

### 5.1 基本評估流程

```python
import mlflow
from mlflow.genai.scorers import Correctness, RelevanceToQuery, Safety

# 準備評估資料
eval_data = [
    {
        "inputs": {"question": "What is MLflow?"},
        "expectations": {
            "expected_facts": ["open-source platform", "ML lifecycle management"]
        },
    },
    {
        "inputs": {"question": "How do I track experiments?"},
        "expectations": {
            "expected_facts": ["mlflow.start_run()", "log metrics"]
        },
    },
]

# 定義預測函式
@mlflow.trace
def my_app(question: str) -> str:
    # 你的 LLM 邏輯
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": question}],
    )
    return response.choices[0].message.content

# 執行評估
results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=my_app,
    scorers=[Correctness(), RelevanceToQuery(), Safety()],
)
```

### 5.2 內建 Scorers

MLflow 提供多種預定義 scorer：

| Scorer | 用途 |
|--------|------|
| `Correctness()` | 檢查是否包含預期事實 |
| `RelevanceToQuery()` | 回答與問題的相關性 |
| `Safety()` | 安全性檢查 |
| `RetrievalGroundedness()` | RAG 回答是否基於檢索內容 |
| `Guidelines(name, guidelines)` | 自定義準則評估 |

### 5.3 Custom Scorer（自定義評分器）

```python
from mlflow.genai.scorers import scorer
from mlflow.entities import Feedback

# 簡單布林 scorer
@scorer
def output_length_check(outputs: str) -> bool:
    """檢查輸出是否超過 100 字元"""
    return len(outputs) > 100

# 返回數值的 scorer
@scorer
def readability_score(outputs: str) -> Feedback:
    import textstat
    score = textstat.flesch_reading_ease(outputs)
    return Feedback(
        value=score,
        rationale=f"Flesch reading ease: {score:.1f}",
    )
```

### 5.4 LLM-as-a-Judge（自定義 LLM 評判）

```python
from typing import Literal
from mlflow.genai.judges import make_judge

# 建立自定義 judge
answer_quality = make_judge(
    name="answer_quality",
    instructions=(
        "Evaluate if the response in {{ outputs }} "
        "correctly and completely answers the question in {{ inputs }}.\n"
        "Return 'yes' if correct, 'no' otherwise."
    ),
    feedback_value_type=Literal["yes", "no"],
    model="openai:/gpt-4o-mini",  # 或 "anthropic:/claude-sonnet-4-20250514"
)

# 用於 evaluate
results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=my_app,
    scorers=[answer_quality],
)
```

### 5.5 Agent-as-a-Judge（Agentic 評估）

使用 `{{ trace }}` 模板變數讓 judge 能主動探索 trace 結構：

```python
from mlflow.genai.judges import make_judge

performance_judge = make_judge(
    name="performance_analyzer",
    instructions=(
        "Analyze the {{ trace }} for performance issues.\n\n"
        "Check for:\n"
        "- Operations taking longer than 2 seconds\n"
        "- Redundant API calls or database queries\n"
        "- Inefficient data processing patterns\n\n"
        "Rate as: 'optimal', 'acceptable', or 'needs_improvement'"
    ),
    model="openai:/gpt-4o",
)

# 直接對單一 trace 評估
trace = mlflow.get_trace(trace_id)
feedback = performance_judge(trace=trace)
print(f"Rating: {feedback.value}, Analysis: {feedback.rationale}")
```

### 5.6 Multi-turn Evaluation（3.7+ / 3.10 強化）

支援多輪對話評估與 session-level scorers：

```python
# 評估現有對話
results = mlflow.genai.evaluate(
    data=conversation_dataset,  # 包含多輪對話的資料集
    scorers=[session_level_scorer],
)
```

3.10 新增 **Conversation Simulation**：無需重新生成對話即可測試 agent 新版本。

### 5.7 MemAlign Judge Optimizer（3.9+）

從歷史 feedback 學習，動態檢索相關範例以提升 judge 準確度：

```python
from mlflow.genai.optimize import MemAlignOptimizer

# 使用 MemAlign 優化 judge
aligned_judge = quality_judge.align(
    traces=historical_traces,
    optimizer=MemAlignOptimizer(),
)
```

### 5.8 Online Monitoring with LLM Judges（3.9+）

在 MLflow UI 的 "Judges" tab 中設定自動評估，無需撰寫程式碼。Traces 到達時自動附加 assessment。

### 5.9 Scorer Registration（持續監控）

```python
from mlflow.genai.scorers import RelevanceToQuery

mlflow.set_experiment("my_genai_app")

# 註冊 scorer 以自動評估新 traces
registered = RelevanceToQuery().register(name="relevance_scorer")
```

---

## 6. AI Gateway（3.9+ 重構）

Gateway 現在直接內嵌於 Tracking Server，提供統一的 LLM API 介面：

```bash
# 不需額外啟動 Gateway 服務
mlflow server --host 0.0.0.0 --port 5000
# Gateway 自動可用
```

支援功能：
- Passthrough endpoints（直接轉發請求）
- Traffic splits（流量分配）
- Fallback models（備援模型）
- Usage tracking（3.10 新增 endpoint 使用量追蹤）

---

## 7. 3.8 – 3.10 重要新功能速覽

### 3.8
- **Prompt Model Configuration**：prompt 可綁定模型設定
- **In-Progress Trace Display**：即時顯示進行中的 trace span
- **DeepEval Judges Integration**：`get_judge` API 可使用 DeepEval 的 20+ 評估指標
- **Conversational Safety Scorer**：多輪對話安全性評估

### 3.9
- **MLflow Assistant**：由 Claude Code 驅動的 in-product debug 助手
- **Trace Overview Dashboard**：GenAI 實驗的「Overview」tab，含延遲、請求量、品質指標
- **Judge Builder UI**：在 UI 中直接定義和迭代 LLM judge prompt
- **Online Monitoring with LLM Judges**：在 "Judges" tab 設定自動評估
- **Distributed Tracing**：跨服務 context propagation
- **MemAlign Optimizer**：新的 judge 優化演算法
- **AI Gateway 重構**：Gateway 內嵌於 Tracking Server

### 3.10
- **Multi-workspace Support**：`--enable-workspaces` 多工作區隔離
- **Multi-turn Evaluation & Conversation Simulation**：對話模擬與 session-level 評估
- **Trace Cost Tracking**：自動計算 LLM 費用，UI 顯示成本趨勢
- **AI Gateway Usage Tracking**：endpoint 使用量追蹤
- **UI 導覽列重新設計**：更清晰的 GenAI vs Classic ML 功能分區

---

## 8. LLM Template 專案建議架構

以下是一個善用 MLflow 功能的 LLM 專案骨架：

```
my_llm_project/
├── pyproject.toml
├── .env                     # API keys & MLflow config
├── src/
│   ├── __init__.py
│   ├── config.py            # MLflow setup & experiment config
│   ├── prompts/
│   │   ├── register.py      # Prompt Registry 管理
│   │   └── templates/       # Prompt 模板原始檔
│   ├── agents/
│   │   ├── base.py          # 基礎 agent 邏輯
│   │   └── langgraph_agent.py  # LangGraph agent
│   ├── evaluation/
│   │   ├── scorers.py       # 自定義 scorers
│   │   ├── judges.py        # 自定義 LLM judges
│   │   ├── datasets.py      # 評估資料集管理
│   │   └── run_eval.py      # 評估執行腳本
│   └── serving/
│       └── app.py           # FastAPI 服務（含 tracing）
├── tests/
│   └── test_evaluation.py   # 評估測試
└── notebooks/
    ├── 01_prompt_engineering.ipynb
    ├── 02_evaluation.ipynb
    └── 03_monitoring.ipynb
```

### 8.1 config.py — MLflow 初始化

```python
import mlflow
import os

def init_mlflow(
    experiment_name: str = "my-llm-app",
    model_name: str = "chatbot-v1",
    tracking_uri: str = "http://localhost:5000",
):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    mlflow.set_active_model(name=model_name)
    
    # 啟用所需的 autolog
    mlflow.openai.autolog()       # 或你使用的 LLM 框架
    mlflow.langchain.autolog()    # 如果使用 LangGraph
    
    return mlflow.get_active_model_id()
```

### 8.2 prompts/register.py — Prompt 管理

```python
import mlflow

def register_or_update_prompt(
    name: str,
    template: str,
    commit_message: str,
    model_config: dict = None,
):
    return mlflow.genai.register_prompt(
        name=name,
        template=template,
        commit_message=commit_message,
        model_config=model_config,
    )

def load_latest_prompt(name: str):
    return mlflow.genai.load_prompt(f"prompts:/{name}@latest")
```

### 8.3 evaluation/scorers.py — 自定義 Scorers

```python
from mlflow.genai.scorers import scorer
from mlflow.genai.judges import make_judge
from mlflow.entities import Feedback
from typing import Literal

# 規則型 scorer
@scorer
def response_not_empty(outputs: str) -> bool:
    return len(outputs.strip()) > 0

@scorer
def response_length_check(outputs: str) -> Feedback:
    length = len(outputs)
    if length < 50:
        return Feedback(value=0.3, rationale="Response too short")
    elif length > 2000:
        return Feedback(value=0.5, rationale="Response too long")
    return Feedback(value=1.0, rationale="Response length is appropriate")

# LLM Judge
tone_judge = make_judge(
    name="professional_tone",
    instructions=(
        "Evaluate if the response maintains a professional tone.\n"
        "Output: {{ outputs }}\n"
        "Return 'yes' if professional, 'no' otherwise."
    ),
    feedback_value_type=Literal["yes", "no"],
    model="openai:/gpt-4o-mini",
)

# Agent-as-a-Judge（分析整個 trace）
efficiency_judge = make_judge(
    name="efficiency_analyzer",
    instructions=(
        "Analyze the {{ trace }} for inefficiencies.\n\n"
        "Check for:\n"
        "- Redundant API calls\n"
        "- Sequential operations that could be parallelized\n"
        "- Unnecessary data processing\n\n"
        "Rate as: 'efficient', 'acceptable', or 'inefficient'"
    ),
    feedback_value_type=Literal["efficient", "acceptable", "inefficient"],
    model="openai:/gpt-4o",
)
```

### 8.4 evaluation/run_eval.py — 執行評估

```python
import mlflow
from mlflow.genai.scorers import Correctness, RelevanceToQuery, Safety
from .scorers import response_not_empty, tone_judge, efficiency_judge

def run_evaluation(predict_fn, eval_data, run_name="eval"):
    with mlflow.start_run(run_name=run_name):
        results = mlflow.genai.evaluate(
            data=eval_data,
            predict_fn=predict_fn,
            scorers=[
                # 內建 scorers
                Correctness(),
                RelevanceToQuery(),
                Safety(),
                # 自定義 scorers
                response_not_empty,
                tone_judge,
            ],
        )
        return results

def run_trace_evaluation(traces, run_name="trace-eval"):
    """對已存在的 traces 執行 Agent-as-a-Judge 評估"""
    with mlflow.start_run(run_name=run_name):
        results = mlflow.genai.evaluate(
            data=traces,
            scorers=[efficiency_judge],
        )
        return results
```

### 8.5 serving/app.py — 帶 Tracing 的 Production 服務

```python
from fastapi import FastAPI
import mlflow
from src.config import init_mlflow
from src.prompts.register import load_latest_prompt

app = FastAPI()
init_mlflow(model_name="chatbot-production")

# 生產環境使用異步 logging
mlflow.config.enable_async_logging(True)

@app.post("/chat")
@mlflow.trace
async def chat(question: str):
    prompt = load_latest_prompt("customer_support_prompt")
    formatted = prompt.format(question=question)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": formatted}],
    )
    
    return {"answer": response.choices[0].message.content}
```

---

## 9. 開發工作流程建議

```
┌─────────────────────────────────────────────────────┐
│  1. Prompt Engineering                              │
│     register_prompt() → load_prompt() → format()    │
│     在 UI Prompts tab 管理版本                        │
├─────────────────────────────────────────────────────┤
│  2. Development + Tracing                           │
│     autolog() 或 @mlflow.trace                      │
│     set_active_model() 連結 traces                   │
│     UI Traces tab 檢視每一步執行細節                    │
├─────────────────────────────────────────────────────┤
│  3. Evaluation                                      │
│     mlflow.genai.evaluate() + built-in scorers      │
│     make_judge() 建立領域專用 LLM judge               │
│     UI Evaluations tab 比較不同版本                    │
├─────────────────────────────────────────────────────┤
│  4. Optimization                                    │
│     optimize_prompts() 自動優化 prompt                │
│     MemAlign 優化 judge 準確度                        │
│     比較 prompt 版本的 evaluation 結果                  │
├─────────────────────────────────────────────────────┤
│  5. Production + Monitoring                         │
│     mlflow-tracing 輕量 SDK                          │
│     Online LLM Judges 自動評估新 traces               │
│     Overview Dashboard 監控延遲、品質、成本             │
│     Distributed Tracing 跨服務追蹤                    │
└─────────────────────────────────────────────────────┘
```

---

## 10. 支援的 Auto-Tracing 框架一覽

| 框架 | 啟用方式 |
|------|---------|
| OpenAI | `mlflow.openai.autolog()` |
| Anthropic | `mlflow.anthropic.autolog()` |
| LangChain / LangGraph | `mlflow.langchain.autolog()` |
| LlamaIndex | `mlflow.llamaindex.autolog()` |
| AutoGen | `mlflow.autogen.autolog()` |
| PydanticAI | `mlflow.pydanticai.autolog()` |
| smolagents | `mlflow.smolagents.autolog()` |
| Semantic Kernel | `mlflow.semantic_kernel.autolog()` |
| DSPy | `mlflow.dspy.autolog()` |
| Agno | `mlflow.agno.autolog()` |
| Gemini (TS) | TypeScript SDK 支援 |

---

## 11. 關鍵 API 速查表

```python
# === Experiment & Model ===
mlflow.set_experiment("my-experiment")
mlflow.set_active_model(name="my-model")
model_id = mlflow.get_active_model_id()

# === Tracing ===
mlflow.openai.autolog()                     # 自動追蹤
@mlflow.trace                               # 手動追蹤
mlflow.start_span("name")                   # 子 span
mlflow.get_last_active_trace_id()           # 取得最新 trace ID
mlflow.get_trace(trace_id)                  # 取得 trace
mlflow.search_traces(filter_string="...")   # 搜尋 traces

# === Prompts ===
mlflow.genai.register_prompt(name, template, commit_message)
mlflow.genai.load_prompt("prompts:/name@latest")
mlflow.genai.search_prompts(filter_string="...")
mlflow.genai.optimize_prompts(predict_fn, train_data, prompt_uris, ...)

# === Evaluation ===
mlflow.genai.evaluate(data, predict_fn, scorers)
mlflow.genai.judges.make_judge(name, instructions, feedback_value_type, model)

# === Scorer Registration (Online Monitoring) ===
scorer_instance.register(name="my_scorer", experiment_id="...")

# === Distributed Tracing ===
from mlflow.tracing.distributed import get_trace_context, set_trace_context
```

---

## 12. 注意事項與最佳實踐

1. **版本要求**：Prompt Optimization 需要 `>=3.5.0`，Distributed Tracing 需要 `>=3.9.0`，Multi-workspace 需要 `>=3.10.0`
2. **predict_fn 必須從 Registry 載入 prompt**：使用 `mlflow.genai.load_prompt()` 而非 hardcode，否則 optimize_prompts 無法運作
3. **Production 使用 mlflow-tracing**：體積僅為完整 mlflow 的 5%，專為低 overhead 設計
4. **異步 logging**：production 環境務必啟用 `enable_async_logging(True)`
5. **Agent-as-a-Judge vs LLM-as-a-Judge**：開發階段用 Agent-as-a-Judge（深度分析），production 用 LLM-as-a-Judge（高效低成本）
6. **LangGraph + Manual Tracing 注意**：混合使用 autolog 與 `@mlflow.trace` 時，確保手動 span 在 autolog 的 trace context 內，避免產生多個獨立 trace
7. **3.10 UI 更新**：導覽列已重新設計，GenAI 與 Classic ML 功能有更清晰的分區

---

*本文件基於 MLflow 3.0 – 3.10.0 官方文件與 release notes 整理，最後更新：2026-03-03*
