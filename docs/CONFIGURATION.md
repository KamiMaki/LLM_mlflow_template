# 設定檔指南

本指南說明如何設定 LLM Framework 的各項參數，以及如何在不同環境之間切換。

## 設定檔格式

設定檔使用 YAML 格式，存放在 `config/` 目錄下。框架支援多個環境（dev、test、stg、prod），每個環境各有一個設定檔。

### 基本結構

```yaml
llm:
  url: "https://api.example.com/v1/chat/completions"
  auth_token: "${LLM_AUTH_TOKEN}"
  default_model: "gpt-4o"
  timeout: 30
  max_retries: 3
  temperature: 0.7

mlflow:
  tracking_uri: "http://mlflow.example.com:5000"
  experiment_name: "my-experiment"
  enabled: true

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

---

## 設定參數一覽

### LLM 設定（`llm`）

| 參數 | 型別 | 必填 | 預設值 | 說明 |
|------|------|------|--------|------|
| `url` | 字串 | 是 | - | LLM API 端點 URL |
| `auth_token` | 字串 | 是 | - | API 認證 Token（建議使用環境變數） |
| `default_model` | 字串 | 否 | `"gpt-4o"` | 預設使用的模型 |
| `timeout` | 整數 | 否 | `30` | 請求逾時秒數 |
| `max_retries` | 整數 | 否 | `3` | 失敗時的最大重試次數 |
| `temperature` | 浮點數 | 否 | `0.7` | 取樣溫度（0.0 = 確定性，1.0 = 創意性） |

### MLflow 設定（`mlflow`）

| 參數 | 型別 | 必填 | 預設值 | 說明 |
|------|------|------|--------|------|
| `tracking_uri` | 字串 | 否 | `""` | MLflow 追蹤伺服器 URI（空字串 = 本地檔案儲存） |
| `experiment_name` | 字串 | 否 | `"default"` | 預設實驗名稱 |
| `enabled` | 布林值 | 否 | `true` | 啟用/停用 MLflow 整合（false = 所有操作變成 no-op） |

### 日誌設定（`logging`）

| 參數 | 型別 | 必填 | 預設值 | 說明 |
|------|------|------|--------|------|
| `level` | 字串 | 否 | `"INFO"` | 日誌等級：DEBUG、INFO、WARNING、ERROR、CRITICAL |
| `format` | 字串 | 否 | 標準格式 | Python logging 格式字串 |

---

## 環境變數解析

框架支援在設定檔中使用 `${VAR_NAME}` 語法來引用環境變數。這對於保護敏感資料（如 API Token）非常重要。

### 語法

```yaml
llm:
  auth_token: "${LLM_AUTH_TOKEN}"    # 從環境變數讀取
  url: "${LLM_API_URL}"             # 也可以用在 URL
```

### 運作方式

1. 設定載入器掃描所有值中的 `${VAR_NAME}` 模式
2. 對每個模式，從系統環境變數中查找對應的值
3. 找到則替換，找不到則拋出 `ConfigError`

### 範例

設定環境變數：

```bash
export LLM_AUTH_TOKEN="sk-abc123def456"
export MLFLOW_TRACKING_URI="http://localhost:5000"
```

載入設定：

```python
from llm_framework.config import load_config

config = load_config("dev")
print(config.llm.auth_token)  # "sk-abc123def456"
```

### 安全最佳實踐

1. **永遠不要將秘密提交到版本控制**
2. **機敏資料一律使用環境變數**
3. **各環境使用不同的 Token**
4. **定期輪替 Token**

---

## 環境切換方式

### 方法一：明確指定

```python
config = load_config("dev")   # 開發環境
config = load_config("prod")  # 生產環境
```

### 方法二：環境變數

```bash
export LLM_ENV=prod
```

```python
config = load_config()  # 自動從 LLM_ENV 讀取
```

### 方法三：執行時切換

```python
for env in ["dev", "stg", "prod"]:
    config = load_config(env)
    print(f"{env}: {config.llm.url}")
```

---

## 各環境範例設定

### 開發環境（`config/dev.yaml`）

```yaml
llm:
  url: "https://internal-llm-dev.company.com/v1/chat/completions"
  auth_token: "${LLM_AUTH_TOKEN}"
  default_model: "gpt-4o"
  timeout: 30
  max_retries: 3
  temperature: 0.7    # 較高溫度，適合創意測試

mlflow:
  tracking_uri: "http://mlflow-dev.company.com:5000"
  experiment_name: "llm-framework-dev"
  enabled: true

logging:
  level: "DEBUG"      # 詳細日誌，方便除錯
```

### 測試環境（`config/test.yaml`）

```yaml
llm:
  url: "https://internal-llm-test.company.com/v1/chat/completions"
  auth_token: "${LLM_AUTH_TOKEN}"
  default_model: "gpt-4o"
  timeout: 30
  max_retries: 3
  temperature: 0.0    # 確定性輸出，便於重現測試

mlflow:
  tracking_uri: ""
  experiment_name: "llm-framework-test"
  enabled: false      # CI 中停用 MLflow

logging:
  level: "WARNING"
```

### Staging 環境（`config/stg.yaml`）

```yaml
llm:
  url: "https://internal-llm-stg.company.com/v1/chat/completions"
  auth_token: "${LLM_AUTH_TOKEN}"
  default_model: "gpt-4o"
  timeout: 45
  max_retries: 3
  temperature: 0.5

mlflow:
  tracking_uri: "${MLFLOW_TRACKING_URI}"
  experiment_name: "llm-framework-stg"
  enabled: true

logging:
  level: "INFO"
```

### 生產環境（`config/prod.yaml`）

```yaml
llm:
  url: "https://internal-llm.company.com/v1/chat/completions"
  auth_token: "${LLM_AUTH_TOKEN}"
  default_model: "gpt-4o"
  timeout: 60         # 較高逾時，確保穩定性
  max_retries: 3
  temperature: 0.0    # 確定性輸出

mlflow:
  tracking_uri: "${MLFLOW_TRACKING_URI}"
  experiment_name: "llm-framework-prod"
  enabled: true

logging:
  level: "WARNING"    # 只記錄警告和錯誤
```

---

## 依賴套件群組

框架使用 `uv` 的 dependency groups 來區分開發和生產環境的依賴。

### 可用群組

| 群組 | 用途 | 安裝指令 |
|------|------|----------|
| `dev` | 完整開發依賴（MLflow、LangGraph、pytest 等） | `uv sync --group dev` |
| `prod` | 僅生產依賴（mlflow-tracing + 必要套件） | `uv sync --group prod` |

### 開發 vs 生產依賴

**開發群組（`dev`）**：
- 完整 MLflow（`mlflow>=2.21`）含 UI 和追蹤伺服器
- LangGraph（`langgraph>=0.3`）工作流程建構
- 測試工具（`pytest`、`pytest-cov`、`pytest-asyncio`）

**生產群組（`prod`）**：
- 僅輕量追蹤（`mlflow-tracing>=2.21`）
- 不包含測試或開發工具
- Docker 映像更小、啟動更快

---

## 進階設定

### 用字典載入設定（測試用）

```python
from llm_framework.config import load_config_from_dict

config = load_config_from_dict({
    "llm": {
        "url": "https://api.example.com",
        "auth_token": "test-token"
    },
    "mlflow": {"enabled": False}
}, env="test")
```

### 存取全域設定

載入一次後，可在任何地方存取：

```python
from llm_framework.config import get_config

config = get_config()
print(config.llm.default_model)
```

### 重設設定（測試用）

```python
from llm_framework.config import reset_config

reset_config()  # 清除全域單例
```

---

## 疑難排解

### 錯誤：「Environment variable 'X' is not set」

**原因**：設定檔中使用了 `${X}` 但環境變數未設定。

**解決**：
```bash
export X="value"
```

### 錯誤：「Config file not found」

**原因**：找不到對應的設定檔（例如 `config/dev.yaml`）。

**解決**：建立設定檔，或指定正確的環境名稱。

### 錯誤：「Configuration not loaded」

**原因**：在呼叫 `get_config()` 之前沒有先呼叫 `load_config()`。

**解決**：在應用程式啟動時先載入設定：
```python
from llm_framework.config import load_config
load_config("dev")  # 務必先執行此行
```
