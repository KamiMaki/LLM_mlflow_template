# LLM Framework — Architecture Design

## Overview

This framework provides a **universal, easy-to-use foundation** for LLM application development. It integrates MLflow for observability and evaluation, LangGraph for workflow orchestration, and supports seamless multi-environment deployment.

**Design Principles:**
1. **Zero boilerplate** — developers focus on prompts and workflow logic, the framework handles everything else
2. **Config-driven** — switch environments, models, and behaviors via YAML config files
3. **Observable by default** — every LLM call is automatically traced and logged
4. **Graceful degradation** — if MLflow is unavailable, the application continues to work
5. **Minimal production footprint** — prod only needs `mlflow-tracing`, not the full MLflow package

---

## Directory Structure

```
llm-framework/
├── pyproject.toml                  # uv project config with dependency groups
├── ARCHITECTURE.md                 # This file
├── README.md                       # Project introduction and quick install
├── config/
│   ├── dev.yaml                    # Development environment config
│   ├── test.yaml                   # Test environment config
│   ├── stg.yaml                    # Staging environment config
│   └── prod.yaml                   # Production environment config
├── src/
│   └── llm_framework/
│       ├── __init__.py
│       ├── config.py               # Config loading & multi-env switching
│       ├── llm_client.py           # Unified LLM call wrapper
│       ├── mlflow/
│       │   ├── __init__.py
│       │   ├── logger.py           # MLflow logging (params, metrics, artifacts)
│       │   ├── tracer.py           # MLflow tracing (spans, auto-trace)
│       │   ├── evaluator.py        # MLflow evaluation (datasets, metrics)
│       │   └── experiment.py       # Experiment & run management
│       ├── workflow/
│       │   ├── __init__.py
│       │   ├── state.py            # Pydantic state models
│       │   ├── base.py             # BaseWorkflow template
│       │   └── tools/
│       │       ├── __init__.py
│       │       ├── parser.py       # JSON parsing with auto-fix
│       │       ├── validator.py    # Pydantic schema output validation
│       │       ├── retry.py        # Exponential backoff retry
│       │       ├── structured_output.py  # Forced-format LLM output
│       │       └── prompt_template.py    # Jinja2-based prompt templates
│       └── utils/
│           ├── __init__.py
│           └── env.py              # Environment variable helpers
├── tests/
│   ├── __init__.py
│   ├── conftest.py                 # Shared fixtures & mocks
│   ├── test_config.py
│   ├── test_llm_client.py
│   ├── test_mlflow/
│   │   ├── __init__.py
│   │   ├── test_logger.py
│   │   ├── test_tracer.py
│   │   ├── test_evaluator.py
│   │   └── test_experiment.py
│   └── test_workflow/
│       ├── __init__.py
│       ├── test_state.py
│       ├── test_base.py
│       └── test_tools.py
├── docs/
│   ├── QUICKSTART.md
│   ├── CONFIGURATION.md
│   ├── API_REFERENCE.md
│   └── FAQ.md
├── examples/
│   ├── 01_hello_world.ipynb
│   ├── 02_single_node_test.ipynb
│   ├── 03_full_workflow.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_multi_env.ipynb
└── Dockerfile
```

---

## Module Specifications

### 1. `config.py` — Configuration Management

**Purpose:** Load YAML config files, resolve environment variables, support multi-environment switching.

**Public API:**
```python
class FrameworkConfig:
    """Immutable configuration object loaded from YAML."""
    llm: LLMConfig
    mlflow: MLflowConfig
    logging: LoggingConfig

class LLMConfig:
    url: str
    auth_token: str
    default_model: str
    timeout: int
    max_retries: int
    temperature: float

class MLflowConfig:
    tracking_uri: str
    experiment_name: str
    enabled: bool

def load_config(env: str = "dev", config_dir: str = "config") -> FrameworkConfig:
    """Load config for the given environment. Resolves ${VAR} placeholders from env vars."""

def get_config() -> FrameworkConfig:
    """Get the currently active config (singleton). Call load_config() first."""
```

**Key Behaviors:**
- `${VAR_NAME}` in YAML values are resolved from environment variables
- Raises `ConfigError` if required env vars are missing
- Config is loaded once and cached as a module-level singleton
- Environment can be set via `LLM_ENV` env var or passed explicitly

---

### 2. `llm_client.py` — Unified LLM Client

**Purpose:** Single interface for all LLM calls. Reads config, handles auth, integrates with MLflow tracing.

**Public API:**
```python
class LLMClient:
    def __init__(self, config: FrameworkConfig | None = None):
        """Initialize with explicit config or use the global singleton."""

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Send a chat completion request. Auto-traced by MLflow."""

    async def achat(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ) -> LLMResponse:
        """Async version of chat()."""

class LLMResponse:
    content: str              # The response text
    model: str                # Model used
    usage: TokenUsage         # Token counts
    latency_ms: float         # Request latency in milliseconds
    raw_response: dict        # Full API response

class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
```

**Key Behaviors:**
- Uses `httpx` for HTTP calls (sync + async)
- Auto-applies auth token from config
- Every call is wrapped in an MLflow trace span (if tracing is enabled)
- Logs token usage and latency automatically
- Respects config timeout and retry settings

---

### 3. `mlflow/logger.py` — MLflow Logging

**Purpose:** Convenient wrappers for logging parameters, metrics, and artifacts to MLflow.

**Public API:**
```python
def log_llm_call(
    prompt: str | list[dict],
    response: str,
    model: str,
    token_usage: dict,
    latency_ms: float,
    params: dict | None = None,
) -> None:
    """Log a single LLM call's details to the active MLflow run."""

def log_params(params: dict) -> None:
    """Log multiple parameters to the active MLflow run."""

def log_metrics(metrics: dict, step: int | None = None) -> None:
    """Log multiple metrics to the active MLflow run."""

def log_artifact(local_path: str, artifact_path: str | None = None) -> None:
    """Log a local file as an MLflow artifact."""

def log_dict_artifact(data: dict, filename: str) -> None:
    """Log a dictionary as a JSON artifact."""
```

**Key Behaviors:**
- All functions are no-ops if MLflow is disabled in config
- Catches MLflow exceptions and logs warnings instead of crashing

---

### 4. `mlflow/tracer.py` — MLflow Tracing

**Purpose:** Span-level tracing for LLM calls and workflow nodes.

**Public API:**
```python
def trace_llm_call(func):
    """Decorator: wraps a function as a traced LLM call span."""

def trace_node(node_name: str):
    """Decorator: wraps a workflow node function as a traced span."""

def trace_workflow(workflow_name: str):
    """Decorator: wraps an entire workflow execution as a parent trace."""

@contextmanager
def span(name: str, span_type: str = "CHAIN") -> Generator:
    """Context manager for creating custom trace spans."""
```

**Key Behaviors:**
- Uses `mlflow.start_span()` / `@mlflow.trace` under the hood
- Automatically captures inputs/outputs of decorated functions
- No-op if MLflow tracing is disabled
- Span types: LLM, CHAIN, TOOL, RETRIEVER (matching MLflow conventions)

---

### 5. `mlflow/evaluator.py` — MLflow Evaluation

**Purpose:** Simplified interface for evaluating LLM outputs.

**Public API:**
```python
class Evaluator:
    def __init__(self, experiment_name: str | None = None):
        """Initialize evaluator, optionally targeting a specific experiment."""

    def evaluate(
        self,
        data: list[dict] | pd.DataFrame,
        metrics: list[str | Callable],
        model_output_col: str = "response",
        target_col: str | None = "expected",
    ) -> EvaluationResult:
        """Run evaluation on a dataset with specified metrics."""

    def compare(
        self,
        run_ids: list[str],
        metric_keys: list[str],
    ) -> pd.DataFrame:
        """Compare metrics across multiple runs."""

class EvaluationResult:
    metrics: dict[str, float]       # Aggregated metric scores
    per_row: pd.DataFrame           # Per-row evaluation details
    run_id: str                     # MLflow run ID for this evaluation
```

**Built-in metric names:** `"correctness"`, `"relevance"`, `"toxicity"`, `"similarity"`, `"latency"`, `"token_efficiency"`

---

### 6. `mlflow/experiment.py` — Experiment Management

**Public API:**
```python
class ExperimentManager:
    def __init__(self, config: FrameworkConfig | None = None):
        """Initialize with config or use global singleton."""

    def get_or_create_experiment(self, name: str | None = None) -> str:
        """Get or create an MLflow experiment. Returns experiment_id."""

    @contextmanager
    def start_run(self, run_name: str | None = None, tags: dict | None = None):
        """Context manager for MLflow runs."""

    def list_runs(
        self, experiment_name: str | None = None, filter_string: str | None = None
    ) -> pd.DataFrame:
        """List runs in an experiment with optional filtering."""

    def get_best_run(self, metric: str, experiment_name: str | None = None) -> dict:
        """Get the run with the best value for a given metric."""
```

---

### 7. `workflow/state.py` — State Models

**Purpose:** Pydantic base models for LangGraph workflow state.

**Public API:**
```python
class BaseState(TypedDict):
    """Minimal state that all workflows share."""
    messages: Annotated[list[dict], add_messages]
    metadata: dict

class LLMState(BaseState):
    """State for single-LLM-call workflows."""
    llm_response: str
    token_usage: dict
    error: str | None

class WorkflowState(BaseState):
    """State for multi-step workflows."""
    current_step: str
    results: dict
    retry_count: int
    error: str | None
```

---

### 8. `workflow/base.py` — BaseWorkflow

**Purpose:** Template for building LangGraph workflows with minimal boilerplate.

**Public API:**
```python
class BaseWorkflow:
    def __init__(self, name: str, state_schema: type = WorkflowState):
        """Create a new workflow with the given name and state type."""

    def add_node(self, name: str, func: Callable, **kwargs) -> "BaseWorkflow":
        """Add a node to the workflow. Returns self for chaining."""

    def add_edge(self, source: str, target: str) -> "BaseWorkflow":
        """Add a direct edge between nodes."""

    def add_conditional_edge(
        self, source: str, condition: Callable, targets: dict[str, str]
    ) -> "BaseWorkflow":
        """Add a conditional routing edge."""

    def set_entry(self, node_name: str) -> "BaseWorkflow":
        """Set the entry point node."""

    def set_finish(self, node_name: str) -> "BaseWorkflow":
        """Set the finish point node."""

    def compile(self) -> CompiledGraph:
        """Compile the workflow into an executable graph."""

    def run(self, input_state: dict, config: dict | None = None) -> dict:
        """Compile (if needed) and run the workflow. Auto-traced."""

    async def arun(self, input_state: dict, config: dict | None = None) -> dict:
        """Async version of run()."""
```

---

### 9. `workflow/tools/` — Workflow Tools

**`parser.py`:**
```python
def parse_json(text: str, fix_common_errors: bool = True) -> dict | list:
    """Parse JSON from LLM output. Optionally fix trailing commas, missing quotes, etc."""

def extract_json(text: str) -> str:
    """Extract JSON block from text that may contain markdown code fences or extra text."""
```

**`validator.py`:**
```python
def validate_output(data: dict, schema: type[BaseModel]) -> BaseModel:
    """Validate LLM output against a Pydantic model. Raises ValidationError on failure."""

def validate_or_none(data: dict, schema: type[BaseModel]) -> BaseModel | None:
    """Validate LLM output, return None instead of raising on failure."""
```

**`retry.py`:**
```python
def with_retry(
    func: Callable,
    max_retries: int = 3,
    backoff_base: float = 1.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple = (Exception,),
) -> Callable:
    """Decorator: retry a function with exponential backoff."""

async def async_with_retry(...) -> Callable:
    """Async version of with_retry."""
```

**`structured_output.py`:**
```python
def get_structured_output(
    client: LLMClient,
    messages: list[dict],
    output_schema: type[BaseModel],
    max_retries: int = 3,
) -> BaseModel:
    """Call LLM and parse response into a Pydantic model. Retries on parse failure."""
```

**`prompt_template.py`:**
```python
class PromptTemplate:
    def __init__(self, template: str):
        """Create a Jinja2-based prompt template."""

    def render(self, **kwargs) -> str:
        """Render the template with the given variables."""

    @classmethod
    def from_file(cls, path: str) -> "PromptTemplate":
        """Load a template from a file."""
```

---

## Integration Points

### LLM Client ↔ MLflow Tracing
Every `LLMClient.chat()` call is automatically wrapped in an MLflow trace span. The span records:
- Input messages
- Output response
- Model name, temperature, token usage, latency

### LangGraph Workflow ↔ MLflow Tracing
`BaseWorkflow.run()` creates a parent trace span. Each node execution creates a child span. This gives full visibility into workflow execution in the MLflow UI.

### Config ↔ Everything
All modules read from the global `FrameworkConfig` singleton. Changing the environment (dev/test/stg/prod) changes all behaviors consistently.

---

## Environment Strategy

| Aspect | dev | test | stg | prod |
|--------|-----|------|-----|------|
| MLflow package | `mlflow` (full) | `mlflow` (full) | `mlflow` (full) | `mlflow-tracing` (light) |
| MLflow tracking | Full logging | Full logging | Full logging | Tracing only |
| LangGraph | Installed | Installed | Installed | Installed |
| Log level | DEBUG | INFO | INFO | WARNING |
| LLM temperature | 0.7 | 0.0 | 0.0 | 0.0 |

---

## Error Handling Strategy

1. **MLflow unavailable:** All MLflow operations are wrapped in try/except. If MLflow server is unreachable, operations become no-ops with warning logs. The LLM workflow continues unaffected.
2. **LLM API errors:** The retry handler provides exponential backoff. After max retries, raises `LLMError` with full context.
3. **Config errors:** Missing required config or env vars raise `ConfigError` at startup, failing fast.
4. **Parse errors:** JSON parsing attempts auto-fix before failing. Schema validation returns clear error messages.
