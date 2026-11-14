# 設定步驟

運作方式：GitHub 每個交易日早上 07:30（台北時間）自動抓行情與新聞、產生今日摘要並存回專案；Streamlit 雲端的網頁會自動讀到最新資料。

## 一、把網頁放上 Streamlit 雲端（只需做一次）

1. 開啟 https://share.streamlit.io ，選 **Continue with GitHub**，用您的 GitHub 帳號登入並同意授權。
2. 右上角 **Create app** → **Deploy a public app from GitHub**。
3. 填寫：
   - Repository：`qwq6666666/us-tw-stock`
   - Branch：`main`
   - Main file path：`streamlit_app.py`
   - App URL：可自訂好記的網址名稱
4. 按 **Deploy**，等二到五分鐘建置完成。

## 二、設成只有自己能看（建議）

1. 在 Streamlit 的 app 清單中，點該 app 右側的「⋮」→ **Settings** → **Sharing**。
2. 「Who can view this app」選 **Only specific people**，填入您自己的電子信箱。
3. 之後在手機或電腦開啟網址時，用該信箱登入即可。免費方案可以有一個私人 app。

若 GitHub 專案改成 Private，Streamlit 第一次存取時會要求額外授權，照畫面同意即可。

## 三、日常使用

- 打開網址就能看。超過 12 小時沒人開啟時 app 會休眠，再次開啟時按畫面上的喚醒按鈕並等幾十秒。
- 可把網址加到手機主畫面。

## 之後想調整

- **換股票清單**：編輯 `core.py` 最上方的 `US_TICKERS`、`TW_TICKERS`。
- **改更新時間**：編輯 `.github/workflows/daily.yml` 的 `cron`（使用 UTC 時間，台北時間減 8 小時）。
- **立刻更新**：到 GitHub 的 Actions 頁面 →「每日更新」→ Run workflow。

## 注意事項

- GitHub 的排程偶爾會延遲幾分鐘到數十分鐘。
- 專案連續 60 天沒有任何更新時，GitHub 會自動停用排程；本專案每天都會存入新資料，正常情況不會發生。
