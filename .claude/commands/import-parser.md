# Import Parser

從其他專案導入現有的 parser / loader，整合進本專案的 dataloader 體系，並生成使用範例。

## 使用方式

```
/import-parser <parser 檔案路徑或目錄>
```

範例：
```
/import-parser ../my-project/parsers/word_parser.py
/import-parser ../my-project/utils/excel_reader.py ../my-project/utils/docx_reader.py
/import-parser ../data-pipeline/loaders/
```

## 執行步驟

### 1. 分析來源 parser

讀取 `$ARGUMENTS` 指定的檔案，分析：

- **輸入格式**：處理什麼檔案類型（.docx, .xlsx, .pdf, .csv 等）
- **輸出格式**：回傳什麼資料結構（dict, list, str, DataFrame 等）
- **依賴套件**：需要什麼第三方套件（python-docx, openpyxl, pandas 等）
- **初始化參數**：是否需要設定檔路徑、API key 等
- **錯誤處理**：有哪些 exception 處理

### 2. 轉換為 BaseLoader 子類別

將 parser 包裝為本專案的 `BaseLoader` 介面：

```python
from app.dataloader.base import BaseLoader
from app.dataloader.models import LoadedData, LoaderConfig


class WordLoader(BaseLoader):
    """從 .docx 檔案載入資料。"""

    def load(self, source: str, **kwargs) -> LoadedData:
        path = self._resolve_path(source)

        # 呼叫原始 parser 邏輯
        content = parse_docx(path)

        return LoadedData(
            content=content,
            source=str(path),
            content_type="docx",
            metadata={"size_bytes": path.stat().st_size},
        )

    def list_sources(self) -> list[str]:
        base = Path(self._config.base_path)
        return [str(p.relative_to(base)) for p in base.rglob("*.docx") if p.is_file()]

    def _resolve_path(self, source: str) -> Path:
        p = Path(source)
        if not p.is_absolute():
            p = Path(self._config.base_path) / p
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        return p
```

### 3. 處理依賴

檢查原始 parser 的依賴，更新 `pyproject.toml`：

```toml
# 在 dependencies 或 dependency-groups 中加入
[dependency-groups]
parsers = [
    "python-docx>=1.0",    # Word 文件解析
    "openpyxl>=3.1",       # Excel 解析
]
```

如果依賴較重（如 pandas），放在 optional dependency group 而非 core dependencies。

### 4. 生成檔案

```
app/dataloader/
├── base.py           # 不修改
├── local.py          # 不修改
├── models.py         # 不修改
├── word.py           # 新增：WordLoader
├── excel.py          # 新增：ExcelLoader
└── __init__.py       # 更新：匯出新 loader
```

### 5. 更新 `__init__.py`

```python
from app.dataloader.base import BaseLoader
from app.dataloader.local import LocalFileLoader
from app.dataloader.models import LoadedData, LoaderConfig
from app.dataloader.word import WordLoader     # 新增
from app.dataloader.excel import ExcelLoader   # 新增
```

### 6. 生成 Workflow 整合範例

展示如何在 workflow 中使用新的 loader 將 raw data 轉為 input state：

```python
from app.dataloader import WordLoader, ExcelLoader
from app.dataloader.models import LoaderConfig

# 初始化 loaders
word_loader = WordLoader(LoaderConfig(base_path="./data/raw"))
excel_loader = ExcelLoader(LoaderConfig(base_path="./data/raw"))

def load_and_prepare_node(state: dict) -> dict:
    """從原始檔案載入資料並轉為 workflow input state。"""
    source_file = state["source_file"]

    if source_file.endswith(".docx"):
        data = word_loader.load(source_file)
    elif source_file.endswith((".xlsx", ".xls")):
        data = excel_loader.load(source_file)
    else:
        raise ValueError(f"Unsupported file type: {source_file}")

    return {
        "input_data": data.content,
        "source": data.source,
        "metadata": data.metadata,
    }
```

### 7. 生成測試

```python
def test_word_loader(tmp_path):
    # 建立測試用 .docx
    ...
    loader = WordLoader(LoaderConfig(base_path=str(tmp_path)))
    result = loader.load("test.docx")
    assert result.content
    assert result.content_type == "docx"

def test_excel_loader(tmp_path):
    ...
```

### 8. 驗證

- 確認新 loader 可以正常 import
- 確認 `BaseLoader` 介面完整實作（`load`, `list_sources`）
- 確認依賴已加入 `pyproject.toml`
- 如果有測試資料，執行一次實際載入驗證

## 注意事項

- **保留原始 parser 的核心邏輯**，只包裝介面，不重寫解析邏輯
- 如果原始 parser 回傳 DataFrame，在 `LoadedData.content` 中存為 `list[dict]`（`.to_dict("records")`）
- 大檔案注意記憶體：考慮 streaming 或分批載入
- 如果原始 parser 有自己的 config，映射到 `LoaderConfig.extra` dict 中
