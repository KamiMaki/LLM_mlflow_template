# 使用指南：AI Service + Trace + Retry

## 1. 敏感資料過濾（Trace Sanitization）

MLflow trace 會自動遮蔽敏感欄位（token、auth、secret、password、api_key 等），不需要額外設定。

**遮蔽規則：**
- 欄位名稱包含 token、auth、secret、password、api_key、credential、bearer 的值會被遮蔽
- 長度 > 8 的字串保留前 4 字元 + `***REDACTED***`
- 短字串直接替換為 `***REDACTED***`

**MLflow trace 會記錄的內容：**
- 遮蔽後的 completion kwargs（含 extra_headers）
- LLM 回應內容（content）
- Reasoning content（如果模型支援）
- Token usage、latency

## 2. 自訂 AI 服務（call_service）

### 設定

在 `llm_config.yaml` 中新增 `service_configs`：

```yaml
service_configs:
  IMAGE_EXTRACTION:
    j1_token: ""  # 或用 LLM_AUTH_TOKEN_IMAGE_EXTRACTION 環境變數
    api_endpoints:
      DEV:  "https://img-extract-dev.internal.com/v1/extract"
      TEST: "https://img-extract-test.internal.com/v1/extract"
    timeout: 60
```

### 呼叫

```python
from llm_service import LLMService

service = LLMService()

# 基本呼叫
result = service.call_service("IMAGE_EXTRACTION", payload={"image": base64_data})
print(result.data)           # 解析後的回應
print(result.status_code)    # HTTP status code
print(result.latency_ms)     # 延遲毫秒

# 自訂回應解析
result = service.call_service(
    "IMAGE_EXTRACTION",
    payload={"image": base64_data},
    response_parser=lambda r: r["data"]["text"],  # 自訂解析邏輯
)
print(result.data)  # "parsed text content"

# 非同步呼叫
result = await service.acall_service("IMAGE_EXTRACTION", payload={...})
```

### 新增自訂服務

1. 在 `llm_config.yaml` 的 `service_configs` 新增設定
2. 使用 `call_service("YOUR_SERVICE_NAME", payload={...})` 呼叫
3. 如果回應格式特殊，傳入 `response_parser` 自訂解析

## 3. J1 Token 讀取

### DEV 環境
在 `llm_config.yaml` 或環境變數中設定 J1 token：
```yaml
model_configs:
  QWEN3:
    j1_token: "your-j1-token"
```
或 `LLM_AUTH_TOKEN_QWEN3=your-j1-token`

### TEST / STG / PROD 環境（Pod）
在 `shared_config` 設定檔案路徑：
```yaml
shared_config:
  j1_token_path: "/var/run/secrets/j1-token"
```

**J1 Token 優先順序：**
1. `LLM_AUTH_TOKEN_<MODEL>` 環境變數（最高）
2. `LLM_AUTH_TOKEN` 環境變數
3. `j1_token` config 值
4. `j1_token_path` 檔案路徑（最低）

## 4. Retry 機制

### 設定

在 `llm_config.yaml` 的 `shared_config.retry` 設定：

```yaml
shared_config:
  retry:
    max_attempts: 3       # 最大重試次數（1 = 不重試）
    wait_multiplier: 1.0  # 指數退避乘數
    wait_min: 2.0         # 最小等待秒數
    wait_max: 10.0        # 最大等待秒數
```

### 行為
- `max_attempts: 1`（預設）= 不重試
- 每次 retry 前會記錄 WARNING log：`Retry attempt 2, waiting 4.0s, error: ConnectionError(...)`
- 使用指數退避（exponential backoff）
- 所有 retry 用完仍失敗時，raise 原始例外
- `call_llm`、`acall_llm`、`call_service`、`acall_service` 均支援
