# 技術文件：專案精簡化 + Skills 設計

## What Changed

### 修改的檔案
| 檔案 | 變更 |
|------|------|
| `app/evaluator/scorers.py` | 5 個 factory → `JUDGE_TEMPLATES` + `create_judge()` |
| `app/prompts/manager.py` | `register()` 移除重複邏輯 |
| `app/prompts/__init__.py` | 移除 `optimize_prompt` import |
| `.gitignore` | 新增 `node_modules/` |

### 刪除的檔案
| 檔案 | 原因 |
|------|------|
| `app/prompts/optimize.py` | GEPA 優化功能過於 niche，依賴特定 MLflow API |
| `app/agents/examples/tool_agent.py` | 範例代碼，不應存在於模板核心 |
| `app/agents/examples/__init__.py` | 同上 |
| `package.json` | Python 專案不需要 Node 依賴 |
| `package-lock.json` | 同上 |

### 新增的檔案
| 檔案 | 用途 |
|------|------|
| `.claude/commands/workflow-build.md` | Workflow 自動建構 skill |
| `.claude/commands/import-parser.md` | Parser 導入 skill |
| `.claude/commands/project-init.md` | 新專案初始化 skill |

## Why

### 精簡化
- **scorers.py**：5 個 judge factory 函式 (`create_tone_judge`, `create_quality_judge`, `create_correctness_judge`, `create_relevance_judge`, `create_safety_judge`) 結構完全相同，唯一差異是 `name` 和 `instructions`。合併為 `JUDGE_TEMPLATES` dict + `create_judge(name)` 入口，減少 ~80 行重複程式碼。
- **manager.py**：`register()` 中 `try` 和 `except` 分支都在建構相同的 `kwargs` dict，合併為一次建構。
- **optimize.py**：依賴 `mlflow.genai.optimize.GepaPromptOptimizer`，這是 MLflow 較新且不穩定的 API，非模板核心功能。
- **agents/examples**：範例代碼應放在 `examples/` notebooks 中，不應在 `app/` 核心目錄。

### Skills 設計
- **`/workflow-build`**：使用者最常見的操作是「描述處理流程 → 得到 workflow 程式碼」。此 skill 封裝了 State 設計、節點生成、驗證邏輯、Graph 組裝的完整流程。
- **`/import-parser`**：使用者通常已有可用的 parser（Word/Excel 等），需要的是快速整合進 dataloader 體系。此 skill 自動處理介面包裝、依賴管理和範例生成。
- **`/project-init`**：結合前兩個 skill，提供「從零到可運行」的完整新專案初始化。

## How It Works

### Scorers 改造

```
Before:
  create_tone_judge()      → create_llm_judge(name="professional_tone", instructions="...")
  create_quality_judge()   → create_llm_judge(name="answer_quality", instructions="...")
  create_correctness_judge() → create_llm_judge(name="correctness", instructions="...")
  ... (5 個幾乎相同的函式)

After:
  JUDGE_TEMPLATES = {
      "professional_tone": "...",
      "answer_quality": "...",
      "correctness": "...",
      "relevance_to_query": "...",
      "safety": "...",
  }
  create_judge("correctness")  # 一個入口搞定
```

### Skills 架構

```
使用者輸入（自然語言描述）
    ↓
Claude Code Command（.claude/commands/*.md）
    ↓
引導 LLM 執行步驟：
  1. 分析需求
  2. 生成程式碼（遵循本專案架構）
  3. 更新設定和依賴
  4. 生成測試
  5. 驗證
```

## Usage

### 精簡後的 Scorers

```python
from app.evaluator.scorers import create_judge, create_llm_judge

# 使用預設範本
judge = create_judge("correctness")
judge = create_judge("safety", temperature=0.0)

# 自訂 judge（與之前相同）
judge = create_llm_judge(
    name="custom",
    instructions="Evaluate...",
)
```

### Skills

```bash
# 建立 workflow
/workflow-build parser → validator → summarizer

# 導入 parser
/import-parser ../my-project/utils/docx_parser.py

# 初始化新專案
/project-init 合約審查系統，Word 和 Excel 輸入
```

## Caveats

- `create_judge()` 只接受 `JUDGE_TEMPLATES` 中定義的名稱，自訂 judge 仍需使用 `create_llm_judge()`
- Skills 是 Claude Code prompt 指令，需要在 Claude Code 環境中使用
- `/project-init` 會建立新的 loader 檔案，需要手動安裝對應的 Python 套件（如 `python-docx`、`openpyxl`）
