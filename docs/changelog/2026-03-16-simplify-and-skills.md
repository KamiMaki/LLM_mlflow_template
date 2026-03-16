# 2026-03-16: 專案精簡化 + 新專案啟動 Skills

## 變更摘要

### 移除的檔案
- `app/prompts/optimize.py` — GEPA prompt optimization（過於 niche，增加複雜度）
- `app/agents/examples/` — 範例 agent（模板中不需要）
- `package.json`, `package-lock.json` — 非必要的 Node.js 依賴
- `__pycache__/` 目錄清理

### 精簡的程式碼
- `app/evaluator/scorers.py` — 5 個重複的 judge factory 函式合併為 `JUDGE_TEMPLATES` dict + `create_judge()` 單一入口
- `app/prompts/manager.py` — `register()` 方法移除重複的 kwargs 建構邏輯
- `app/prompts/__init__.py` — 移除已刪除的 `optimize_prompt` import

### 新增的 Skills（Claude Code Commands）
- `/workflow-build` — 描述 nodes + 連接方式，自動生成 LangGraph workflow（含驗證節點）
- `/import-parser` — 從其他專案導入 parser/loader，整合進 dataloader 體系
- `/project-init` — 新專案初始化，設定完整 data pipeline

### 其他
- `.gitignore` 新增 `node_modules/`
