# Technical Doc: AI Service + Trace Sanitization + Retry

## What Changed

### New Files
- `llm_service/trace.py` — 敏感資料過濾 + MLflow span context manager

### Modified Files
- `llm_service/config.py` — RetryConfig, ServiceConfig, j1_token_path, resolve_service
- `llm_service/models.py` — AIServiceResponse dataclass
- `llm_service/service.py` — retry wrapper, trace span, call_service/acall_service
- `llm_service/__init__.py` — 新增 exports
- `app/logger/setup.py` — litellm autolog log_traces=False
- `llm_config.yaml` — j1_token_path, retry, service_configs
- `tests/test_llm_service.py` — 84 tests (38 new)

## Why

1. **Trace Sanitization**: MLflow litellm autolog 會記錄 `extra_headers` 中的 token，造成機密外洩。改為手動建立 sanitized span，並加入 `reasoning_content`。
2. **AI Service**: 公司內部有非 LLM 的 AI 服務（圖片辨識等），共用 J1→J2 auth 但回應格式不同，需要可擴充的 service 架構。
3. **J1 Token Path**: DEV 從 config 讀取，TEST/PROD 從 pod 掛載路徑讀取。
4. **Retry**: 網路呼叫需要重試機制，避免暫時性錯誤導致流程中斷。

## How It Works

### Trace Architecture

```
LLMService.call_llm()
  │
  ├─ sanitize_completion_kwargs(kwargs)  ← 遮蔽 extra_headers 中的 token
  │
  ├─ trace_span("LLMService.call_llm")  ← MLflow span (sanitized inputs)
  │   │
  │   ├─ _execute_with_retry(litellm.completion, ...)
  │   │   └─ tenacity Retrying (configurable)
  │   │
  │   └─ span.set_outputs({content, reasoning_content, usage, latency})
  │
  └─ return LLMResponse
```

**Key Design**: `mlflow.litellm.autolog(log_traces=False)` 停用自動 trace，由 `trace_span()` 手動建立包含 sanitized data 的 span。

### Sensitive Data Filtering

```
trace.py
├─ SENSITIVE_KEY_PATTERNS: tuple[re.Pattern]  ← token, auth, secret, password...
├─ is_sensitive_key(key) → bool
├─ sanitize_dict(data) → dict                 ← 遞迴遮蔽
├─ sanitize_completion_kwargs(kwargs) → dict   ← 針對 litellm kwargs
└─ trace_span(name, inputs) → context manager  ← MLflow span helper
```

### Config Model Relationships

```
LLMConfig
├─ shared_config: SharedConfig
│   ├─ j1_token_path: str          ← NEW: pod token path
│   └─ retry: RetryConfig          ← NEW: tenacity config
│       ├─ max_attempts: int
│       ├─ wait_multiplier: float
│       ├─ wait_min: float
│       └─ wait_max: float
├─ model_configs: dict[str, ModelConfig]       ← LLM models
└─ service_configs: dict[str, ServiceConfig]   ← NEW: custom AI services
    ├─ j1_token: str
    ├─ api_endpoints: dict[str, str]
    ├─ timeout: int
    └─ extra_headers: dict[str, str]
```

### J1 Token Resolution Flow

```
_resolve_j1_token(alias, token_from_config)
  1. os.getenv("LLM_AUTH_TOKEN_{alias}")   ← env per-model/service
  2. os.getenv("LLM_AUTH_TOKEN")           ← env fallback
  3. token_from_config                      ← yaml j1_token
  4. _read_j1_from_file()                   ← shared_config.j1_token_path
```

### Retry Mechanism

使用 tenacity 的 `Retrying` / `AsyncRetrying`：

```python
# Sync
retryer = Retrying(
    stop=stop_after_attempt(cfg.max_attempts),
    wait=wait_exponential(multiplier=..., min=..., max=...),
    before_sleep=_log_before_retry,  # loguru WARNING
    reraise=True,
)
return retryer(fn, *args, **kwargs)
```

- `max_attempts=1`: 直接呼叫，不包裝 tenacity
- `max_attempts>1`: 使用 exponential backoff retry
- 每次 retry 前記錄 WARNING log

### call_service Flow

```
LLMService.call_service(service_name, payload, response_parser)
  │
  ├─ config.resolve_service(name, zone)
  │   ├─ get_service_config(name) → ServiceConfig
  │   ├─ get_api_endpoint(zone) → endpoint URL
  │   ├─ _resolve_j1_token() → J1
  │   └─ _exchange_token() → J2
  │
  ├─ sanitize_dict(headers + payload) for trace
  │
  ├─ trace_span("AIService.{name}")
  │   ├─ httpx.Client.post(endpoint, json=payload, headers=...)
  │   └─ response_parser(raw) if provided
  │
  └─ return AIServiceResponse(data, status_code, latency_ms, raw_response)
```

## Usage

```python
from llm_service import LLMService

service = LLMService()

# LLM call (with retry + sanitized trace)
resp = service.call_llm(user_prompt="Hello")
print(resp.reasoning_content)  # reasoning model 的思考過程

# Custom AI service
result = service.call_service(
    "IMAGE_EXTRACTION",
    payload={"image": base64_data},
    response_parser=lambda r: r["data"]["text"],
)
```

## Caveats

- `mlflow.litellm.autolog(log_traces=False)` 需要 MLflow 3.x+；舊版會 fallback 到一般 autolog
- `trace_span()` 在 MLflow 未安裝時自動跳過（graceful degradation）
- Retry 的 `wait_min`/`wait_max` 單位為秒，建議 production 設 `max_attempts: 3`
- `call_service` 使用 httpx（非 litellm），每次呼叫建立新的 httpx.Client
- `j1_token_path` 讀取失敗時靜默回傳空字串，不會 raise exception
