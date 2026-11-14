# 設定步驟

完成後的運作方式：GitHub 每個交易日早上 07:30（台北時間）自動抓行情與新聞、產生今日摘要，再把結果送到 Hugging Face 的私人網頁。您只要用手機或電腦打開網址。

## 一、Hugging Face（網頁放置處）

1. 到 https://huggingface.co/join 註冊帳號並完成信箱驗證。記下您的使用者名稱。
2. 右上角頭像 → Settings → Access Tokens → Create new token。
   - Token type 選 **Write**，名稱隨意（例如 `github-deploy`）。
   - 建立後複製 `hf_` 開頭的字串。它只會顯示一次，請先貼到記事本。

不需要自己建立 Space，第一次排程執行時會自動建立一個私人的 Space。

## 二、GitHub（排程與資料存放處）

1. 進入這個專案 → Settings → Secrets and variables → Actions → New repository secret，新增兩筆：

   | Name | Secret |
   |---|---|
   | `HF_TOKEN` | 上一步複製的 `hf_...` 字串 |
   | `HF_SPACE` | `您的HF使用者名稱/us-tw-stock`（例如 `alice/us-tw-stock`） |

2. 上方 Actions 分頁 → 若出現「I understand my workflows...」按鈕，按下啟用。
3. 左側選「每日更新」→ 右側 Run workflow → Run workflow，手動執行第一次。
4. 等一到三分鐘，圓圈變成綠色勾勾即成功。若是紅色叉叉，點進去把錯誤訊息複製下來。

## 三、開啟網頁

1. 到 `https://huggingface.co/spaces/您的HF使用者名稱/us-tw-stock`，第一次需要等幾分鐘建置。
2. 這是私人頁面，手機上也要先登入 Hugging Face 才看得到。可把網址加到手機主畫面。

## 之後想調整

- **換股票清單**：編輯 `core.py` 最上方的 `US_TICKERS`、`TW_TICKERS`。
- **改更新時間**：編輯 `.github/workflows/daily.yml` 的 `cron`（使用 UTC 時間，台北時間減 8 小時）。
- **立刻更新**：到 Actions 頁面手動 Run workflow。

## 注意事項

- GitHub 的排程偶爾會延遲幾分鐘到數十分鐘。
- 專案連續 60 天沒有任何更新時，GitHub 會自動停用排程；本專案每天都會存入新資料，正常情況不會發生。
- `HF_TOKEN` 等同密碼，請只貼在 GitHub 的 Secrets 欄位，不要貼到其他地方。
