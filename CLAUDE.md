# 專案說明（給 Claude Code）

美股 × 台股整合分析的個人網頁。使用者是股市初學者，會照步驟操作但不寫程式；請用繁體中文、白話說明，專有名詞要解釋，並如實說明哪些部分已驗證、哪些還沒。所有輸出都是研究輔助，不構成投資建議。

## 架構

- `core.py`：股票清單設定（檔案最上方）、指標計算（`Market` 類別）、白話摘要（`build_summary`、`describe_stock`）、新聞蒐集。
- `fundamentals.py`：公司體質。台股抓證交所 OpenAPI（`t187ap05_L` 月營收、`t187ap06_L_ci` 損益表、`t187ap07_L_ci` 資產負債表、`BWIBBU_ALL` 本益比／殖利率、`t187ap03_L` 產業別），美股用 yfinance；算指標、五面向評分（固定門檻）、同業比較，存成 `data/fundamentals.json`。
- `model.py`：漲跌機率模型（LightGBM，台股、美股各一個；預測 5／10／20 個交易日後上漲機率；walk-forward 時間切分驗證並與基準比較；SHAP 式影響因素）→ `data/model.json`。
- `scorecard.py`：模型成績單。每天把判斷追加到 `data/predictions_log.csv`（只新增、SHA-256 雜湊鏈防竄改，同（判斷日,標的,期間）只記第一次）；到期用 `prices.csv` 的收盤價對答案，算累計／近期命中率、Brier、機率校準。`MODEL_VER` 在模型設定有實質改動時要換。
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

1. ✅ **「公司體質」分頁（已完成 2026-11-15，台股與美股資料已用真實來源實測、網頁以 AppTest 測過；尚未在 Streamlit Cloud 與 GitHub Actions 上驗證）**：成長、獲利能力、財務體質、股東回報、估值五個面向評分，加白話結論與同業比較。台股資料來自證交所 OpenAPI（月營收、財報彙總），美股來自 yfinance。先以關注清單內的個股設計。已知限制：台股獲利年增率暫缺（證交所只有當期累計，沒有去年同期）；ROE 以累計淨利年化估算；美股同業僅限關注清單；金融業的負債比不適用。
2. ✅ **漲跌機率模型（已完成 2026-11-15，但驗證結果誠實地「沒勝過基準」）**：價量、美股隔夜、匯率特徵，按時間切分驗證，台股／美股 × 1／2／4 週共 6 組，AUC 都在 0.44～0.50、Brier 皆不優於「猜歷史上漲比例」。試過 7／3／2 葉三種複雜度仍然如此，之後選最保守的一組，沒有再針對驗證期調參。網頁「漲跌機率」分頁會明確警示，並附「體質 × 趨勢」四格表。**尚未做**：法人買賣超特徵（證交所 OpenAPI 沒有歷史資料，需先逐日累積）；新聞情緒（第 4 項）。尚未在 GitHub Actions／Streamlit 雲端驗證。
3. ✅ **「模型成績單」分頁（已完成 2026-11-15）**：從 2026-11-13 的判斷開始記錄（51 筆，不回填歷史）。1 週期最快 2026-11-20 之後到期；目前已到期筆數為 0，成績單先顯示「樣本太少」警示。模型每次每日更新都用截至當天的資料重新訓練（等同每日重訓）。程式邏輯（寫入、去重、竄改偵測、對答案）已用測資驗證，網頁以 AppTest 測過空資料與 200 筆模擬資料；尚未在 GitHub Actions 驗證排程會正確追加。**注意**：`data/predictions_log.csv` 絕對不要手動編輯或重建，否則雜湊鏈會斷；排程若因 git 衝突失敗，要保留舊紀錄。
4. **新聞情緒模型**：新聞累積數週後，在 Colab（使用者有 Google AI Pro，含 Colab 運算單元）以小模型加 LoRA 微調，或先用 BERT 類分類器當基準；情緒分數併入第 2 項的特徵並檢驗是否有幫助。
5. **語言模型閱讀法說會與公告**，補上前景分析的文字面；「今日摘要」之後可由規則產生升級為 AI 撰寫。

## 尚未處理

- GitHub 專案與 Streamlit app 目前都是公開的；使用者原本希望只有自己使用，已告知如何改為私人。
- GitHub Secrets 裡的 `HF_TOKEN`、`HF_SPACE` 已不再使用，可刪除。
