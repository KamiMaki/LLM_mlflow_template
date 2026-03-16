# Skills 使用指南

本專案提供三個 Claude Code Skills，用於快速啟動新的 LLM 專案。

## `/workflow-build` — 自動建立 Workflow

告訴 LLM 你需要哪些 nodes 和連接方式，自動生成完整的 LangGraph workflow。

**使用方式：**
```
/workflow-build 三個節點：parser 解析輸入 → checker 檢查格式 → summarizer 摘要輸出
```

**特色：**
- 每個 LLM 節點自動加入驗證邏輯
- 驗證失敗時自動重試（最多 3 次）
- 自動生成對應的 State 定義和測試

---

## `/import-parser` — 導入現有 Parser

從其他專案導入已寫好的 parser / loader，自動包裝為 `BaseLoader` 子類別。

**使用方式：**
```
/import-parser ../my-project/parsers/word_parser.py
/import-parser ../data-pipeline/loaders/
```

**支援場景：**
- Word (.docx) → 提取段落、表格
- Excel (.xlsx) → 讀取 sheet 轉為 list[dict]
- 任何自訂格式的 parser

**自動處理：**
- 包裝為 BaseLoader 介面
- 更新 pyproject.toml 依賴
- 生成整合範例和測試

---

## `/project-init` — 新專案初始化

從模板初始化一個完整的 LLM 專案，包含 data pipeline。

**使用方式：**
```
/project-init 合約審查系統，從 Word 合約和 Excel 條件表開始
/project-init 客服品質分析，讀取對話紀錄 Excel 用 LLM 評分
```

**自動生成：**
- Data pipeline（raw files → parser → input state）
- LangGraph workflow（含驗證節點）
- 評估支援（make_workflow_predict_fn）
- 測試案例
- 設定檔更新

---

## 組合使用

典型的新專案啟動流程：

```
1. /project-init 描述你的專案需求
2. /import-parser 導入現有的 parser（如果有）
3. /workflow-build 細調 workflow 節點
4. /migrate 移植舊有的 workflow（如果有）
```
