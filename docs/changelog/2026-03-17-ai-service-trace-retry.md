# Changelog: AI Service + Trace Sanitization + Retry

**Date:** 2026-03-17

## What Changed

### New Features

1. **MLflow Trace Sensitive Data Filtering**
   - `llm_service/trace.py` (NEW): 敏感資料過濾工具，自動遮蔽 token、auth、secret 等欄位
   - `app/logger/setup.py`: 停用 litellm autolog traces，改由 LLMService 手動建立 sanitized span
   - MLflow trace 現在會記錄 `reasoning_content`（reasoning model 的思考過程）

2. **Custom AI Service Support (call_service)**
   - `llm_service/config.py`: 新增 `ServiceConfig` 模型，支援自訂 AI 服務端點配置
   - `llm_service/service.py`: 新增 `call_service()` / `acall_service()` 方法
   - `llm_service/models.py`: 新增 `AIServiceResponse` 資料模型
   - 支援自訂 `response_parser` 函式解析不同服務的回應格式

3. **J1 Token File-Based Reading**
   - `llm_service/config.py`: SharedConfig 新增 `j1_token_path` 欄位
   - 非 DEV 環境可從 pod 掛載路徑讀取 J1 token
   - 優先順序：env var > config value > file path

4. **Tenacity Retry Mechanism**
   - `llm_service/config.py`: 新增 `RetryConfig` 模型（max_attempts, wait_multiplier, wait_min, wait_max）
   - `llm_service/service.py`: `call_llm` / `acall_llm` / `call_service` / `acall_service` 均支援自動重試
   - 每次 retry 都有 loguru WARNING 記錄

### Modified Files

| File | Change |
|------|--------|
| `llm_service/trace.py` | NEW — 敏感資料過濾 + MLflow span helper |
| `llm_service/config.py` | 新增 RetryConfig, ServiceConfig, j1_token_path, resolve_service() |
| `llm_service/models.py` | 新增 AIServiceResponse |
| `llm_service/service.py` | 新增 retry, trace, call_service, acall_service |
| `llm_service/__init__.py` | 更新 exports |
| `app/logger/setup.py` | litellm autolog log_traces=False |
| `llm_config.yaml` | 新增 j1_token_path, retry, service_configs 範例 |
| `tests/test_llm_service.py` | 新增 38 個測試（共 84 個，全數通過） |
