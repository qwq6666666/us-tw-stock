# 美股 × 台股 整合分析

每天台股開盤前自動更新的個人分析網頁：今日摘要、個股查詢、新聞、績效比較、美股對台股的連動、ADR 溢價。

- `core.py`：設定（股票清單）、計算與新聞蒐集
- `update.py`：每日更新，結果存到 `data/`
- `streamlit_app.py`：網頁介面
- `.github/workflows/daily.yml`：每日排程
- `SETUP.md`：設定步驟

僅供研究與學習，不構成投資建議。
