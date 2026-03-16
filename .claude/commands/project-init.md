# Project Init

從本模板初始化一個新的 LLM 專案，設定完整的 data pipeline：raw files → parser → input state → workflow → output。

## 使用方式

```
/project-init <專案描述>
```

範例：
```
/project-init 合約審查系統，從 Word 合約和 Excel 條件表開始，解析後檢查合規性
/project-init 客服品質分析，讀取客服對話紀錄（Excel），用 LLM 評分並生成報告
```

## 執行步驟

### 1. 需求分析

從 `$ARGUMENTS` 提取：

- **資料來源**：什麼格式的檔案（Word, Excel, PDF, CSV 等）
- **處理流程**：需要哪些處理步驟（解析、檢查、分析、摘要等）
- **輸出目標**：最終產物是什麼（報告、分數、結構化資料等）

如果描述不夠明確，**主動詢問**：
- 輸入檔案的具體格式和欄位？
- 需要哪些 LLM 處理步驟？
- 輸出格式和目標？

### 2. 設定 Data Pipeline

#### 2a. 識別需要的 parsers

根據資料來源，決定需要哪些 loader：

| 檔案類型 | Loader | 依賴套件 |
|---------|--------|---------|
| .docx | WordLoader | python-docx |
| .xlsx/.xls | ExcelLoader | openpyxl |
| .pdf | PdfLoader | pypdf |
| .csv/.json/.txt | LocalFileLoader | （內建） |

#### 2b. 建立 parsers

如果使用者有現成的 parser：
- 詢問 parser 檔案路徑
- 使用 `/import-parser` 的邏輯導入

如果沒有現成 parser：
- 根據檔案格式建立新的 loader（繼承 `BaseLoader`）
- Word：提取段落文字、表格內容
- Excel：讀取指定 sheet，轉為 `list[dict]`

#### 2c. 建立資料轉換層

在 `app/workflow/` 中建立 `data_pipeline.py`：

```python
from app.dataloader import WordLoader, ExcelLoader, LocalFileLoader
from app.dataloader.models import LoaderConfig

# 根據設定初始化 loaders
loaders = {
    ".docx": WordLoader(LoaderConfig(base_path="./data/raw")),
    ".xlsx": ExcelLoader(LoaderConfig(base_path="./data/raw")),
    ".csv": LocalFileLoader(LoaderConfig(base_path="./data/raw")),
}

def load_raw_data(file_path: str) -> dict:
    """載入原始檔案並轉為統一的 dict 格式。"""
    suffix = Path(file_path).suffix.lower()
    if suffix not in loaders:
        raise ValueError(f"No loader for {suffix}. Available: {list(loaders.keys())}")

    data = loaders[suffix].load(file_path)
    return {
        "raw_content": data.content,
        "source": data.source,
        "content_type": data.content_type,
        "metadata": data.metadata,
    }
```

### 3. 設計 Workflow

使用 `/workflow-build` 的邏輯，根據處理流程建立 LangGraph workflow。

典型流程：

```
START → load_data → parse_content → [validate] → process_with_llm → [validate] → format_output → END
```

每個節點：
- `load_data`：呼叫 data_pipeline.load_raw_data()
- `parse_content`：將 raw content 轉為結構化的 input state
- `process_with_llm`：LLM 處理核心邏輯（檢查/分析/摘要）
- `format_output`：將結果格式化為最終輸出

### 4. 生成完整專案結構

```
app/
├── workflow/
│   ├── __init__.py           # 匯出 workflow
│   ├── state.py              # State 定義
│   ├── data_pipeline.py      # 資料載入 + 轉換
│   └── <project_name>.py     # 主 workflow
├── dataloader/
│   ├── ... (既有)
│   ├── word.py               # WordLoader（如需要）
│   └── excel.py              # ExcelLoader（如需要）
data/
├── raw/                      # 放原始檔案
│   └── .gitkeep
├── eval/
│   └── test_cases.json       # 評估測試案例
```

### 5. 生成範例資料和測試案例

#### 範例 test_cases.json：

```json
[
    {
        "inputs": {"source_file": "example.docx"},
        "expectations": {"expected": "包含合規性分析結果"}
    }
]
```

#### 範例測試：

```python
def test_data_pipeline():
    result = load_raw_data("test_file.docx")
    assert "raw_content" in result

def test_workflow_end_to_end():
    result = compiled_graph.invoke({"source_file": "test_file.docx"})
    assert result["final_output"]
```

### 6. 更新設定

#### config/config.yaml — 加入新的 dataloader 設定：

```yaml
dataloader:
  base_path: "./data/raw"
  encoding: "utf-8"
```

#### pyproject.toml — 加入需要的依賴：

```toml
[dependency-groups]
parsers = [
    "python-docx>=1.0",
    "openpyxl>=3.1",
]
```

### 7. 生成 README 區段

在 `README.md` 中加入：

```markdown
## Quick Start

1. 將原始檔案放入 `data/raw/`
2. 執行 workflow：
   ```python
   from app.workflow import compiled_graph
   result = compiled_graph.invoke({"source_file": "your_file.docx"})
   print(result["final_output"])
   ```
```

### 8. 驗證

- 確認所有檔案可正確 import
- 確認 data pipeline 可載入測試資料
- 確認 workflow graph 可正確 compile
- 執行 `python -m pytest tests/ -x` 確認無錯誤

## 設計原則

- **Data pipeline 與 workflow 分離**：載入/解析是獨立模組，workflow 只處理業務邏輯
- **統一入口**：所有原始資料都經過 `load_raw_data()` → 統一 dict 格式 → workflow state
- **可擴充**：新增檔案格式只需加新 loader，不需改 workflow
- **每個 LLM 節點都有驗證**：自動加入格式和內容檢查
