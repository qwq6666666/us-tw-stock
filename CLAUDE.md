# 專案說明（給 Claude Code）

美股 × 台股整合分析的個人網頁。使用者是股市初學者，會照步驟操作但不寫程式；請用繁體中文、白話說明，專有名詞要解釋，並如實說明哪些部分已驗證、哪些還沒。所有輸出都是研究輔助，不構成投資建議。

## 架構

- `core.py`：股票清單設定（檔案最上方）、指標計算（`Market` 類別）、白話摘要（`build_summary`、`describe_stock`）、新聞蒐集。
- `update.py`：每日更新。下載行情 → `data/prices.csv`；新聞累積 → `data/news_archive.csv`；摘要 → `data/summary.md`；狀態 → `data/meta.json`。
- `streamlit_app.py`：網頁介面，只讀 `data/`；「個股查詢」遇到清單外的代號才即時向 Yahoo Finance 查詢。
- `.github/workflows/daily.yml`：台北時間週一至週五 07:30 執行 `update.py` 並把 `data/` 存回 `main`。
- 網頁：https://us-tw-stock.streamlit.app （Streamlit Community Cloud，`main` 有新 commit 就自動同步）。

## 已確認可用（2026-11-15）

- 每日排程全流程、四個新聞來源（鉅亨網台股／美股、證交所重大訊息、Yahoo Finance）。
- 網頁六個分頁，以及清單外代號的即時查詢。

## 開發注意事項

- `data/` 由排程寫入，改程式前先 `git pull`，不要手動編輯裡面的檔案。
- 介面可用 `streamlit.testing.v1.AppTest` 對 `data/` 內的真實資料做無頭測試。
- 開發環境若連不到 Yahoo／鉅亨網／證交所，抓資料的程式要靠 GitHub Actions 驗證：
  `gh api -X POST repos/qwq6666666/us-tw-stock/actions/workflows/daily.yml/dispatches -f ref=main`
- 新聞來源都不是官方保證的介面，格式可能變動；每個來源各自 try/except，失敗不影響其他來源。
- Hugging Face Spaces 的 Gradio/Docker 已改為付費，不要再提議；Streamlit 免費方案只有一個私人 app。
- 「美股隔夜 → 台股隔日」只能用嚴格早於該台股交易日的美股收盤價（見 `Market.us_known`），新增特徵時要守住這條，避免偷看未來資料。
- ADR 換股比例：TSM 5、UMC 5、ASX 2、CHT 10。

## 待辦（依序）

1. **「公司體質」分頁**：成長、獲利能力、財務體質、股東回報、估值五個面向評分，加白話結論與同業比較。台股資料來自證交所 OpenAPI（月營收、財報彙總），美股來自 yfinance。先以關注清單內的個股設計。
2. **漲跌機率模型**：梯度提升樹（LightGBM 類），預測未來一到四週上漲機率；特徵含趨勢、動能、美股隔夜、匯率、法人買賣超。必須按時間切分驗證，並與「永遠猜漲」等簡單基準比較。輸出機率與主要影響因素，不給買賣結論。搭配「體質好壞 × 趨勢多空」四格判讀表。
3. **「模型成績單」分頁**：每天把判斷寫入不可修改的紀錄，到期後對答案，顯示累計與近期命中率、機率校準；排程內定期重新訓練。
4. **新聞情緒模型**：新聞累積數週後，在 Colab（使用者有 Google AI Pro，含 Colab 運算單元）以小模型加 LoRA 微調，或先用 BERT 類分類器當基準；情緒分數併入第 2 項的特徵並檢驗是否有幫助。
5. **語言模型閱讀法說會與公告**，補上前景分析的文字面；「今日摘要」之後可由規則產生升級為 AI 撰寫。

## 尚未處理

- GitHub 專案與 Streamlit app 目前都是公開的；使用者原本希望只有自己使用，已告知如何改為私人。
- GitHub Secrets 裡的 `HF_TOKEN`、`HF_SPACE` 已不再使用，可刪除。
