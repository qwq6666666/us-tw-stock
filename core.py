"""美股 × 台股整合分析：共用的設定、計算與新聞蒐集邏輯。"""
import os
import numpy as np
import pandas as pd

# ───────────── 設定（要換股票改這裡） ─────────────
# 美股直接用代號；台股上市加 .TW、上櫃加 .TWO；指數以 ^ 開頭
US_TICKERS = {
    '^GSPC': 'S&P 500',
    '^IXIC': '那斯達克',
    '^SOX':  '費城半導體',
    'NVDA':  '輝達',
    'AAPL':  '蘋果',
}
TW_TICKERS = {
    '^TWII':   '加權指數',
    '2330.TW': '台積電',
    '2317.TW': '鴻海',
    '2454.TW': '聯發科',
    '0050.TW': '元大台灣50',
}
# ADR 代號 -> (台股代號, 1 單位 ADR 代表幾股, 名稱)
ADR_PAIRS = {
    'TSM': ('2330.TW', 5,  '台積電'),
    'UMC': ('2303.TW', 5,  '聯電'),
    'ASX': ('3711.TW', 2,  '日月光投控'),
    'CHT': ('2412.TW', 10, '中華電信'),
}
START = '2010-01-01'      # 2026-11-17 由 2021 改為 2010：個股驗證期從約 5 年延長到約 15 年（涵蓋 2011、2015、2018、2020、2022 的下跌）
FX_TICKER = 'TWD=X'          # 1 美元兌多少新台幣
US_LEAD = '^SOX'             # 用哪個美股指標觀察對台股的影響
TW_TARGET = '^TWII'
TRADING_DAYS = 252
# ────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PRICES_CSV = os.path.join(DATA_DIR, 'prices.csv')
NEWS_CSV = os.path.join(DATA_DIR, 'news_archive.csv')
SUMMARY_MD = os.path.join(DATA_DIR, 'summary.md')
META_JSON = os.path.join(DATA_DIR, 'meta.json')

NAMES = {**US_TICKERS, **TW_TICKERS}
for _adr, (_tw, _, _nm) in ADR_PAIRS.items():
    NAMES.setdefault(_adr, _nm + ' ADR')
    NAMES.setdefault(_tw, _nm)

US_COLS = list(dict.fromkeys(list(US_TICKERS) + list(ADR_PAIRS)))
TW_COLS = list(dict.fromkeys(list(TW_TICKERS) + [v[0] for v in ADR_PAIRS.values()]))


def label(t):
    return f'{NAMES[t]} ({t})' if t in NAMES else t


def clean_breaks(df, limit_tw=0.15, limit_us=0.6):
    """資料斷點防護（2010 起的長歷史才需要）：單日漲跌幅超過門檻，視為資料錯誤或股票重整（未還原的分割、代號被重複使用、分割出新公司），
    該檔「最後一個斷點之前」的歷史全部丟掉，只保留之後連續的資料。台股漲跌幅限制 7%（2015 年起 10%），門檻取 15%；美股取 60%。
    指數與匯率不處理。回傳（清理後的表, [(代號, 斷點日期), ...]）。規則在看結果前固定。"""
    out, cut = df.copy(), []
    for c in out.columns:
        if str(c).startswith('^') or str(c).endswith('=X'):
            continue
        r = out[c].pct_change(fill_method=None).abs()
        lim = limit_tw if str(c).endswith(('.TW', '.TWO')) else limit_us
        bad = r[r > lim]
        if len(bad):
            d = bad.index[-1]
            out.loc[out.index < d, c] = float('nan')
            cut.append((c, d.strftime('%Y-%m-%d')))
    return out, cut


def download_close(tickers, start=START):
    """下載還原權息後的收盤價，回傳以日期為索引的表格。"""
    import yfinance as yf
    raw = yf.download(list(tickers), start=start, auto_adjust=True, progress=False, group_by='column')
    close = raw['Close']
    if isinstance(close, pd.Series):
        close = close.to_frame(list(tickers)[0])
    close = close.copy()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.index.name = 'Date'
    close, _ = clean_breaks(close)
    return close.dropna(how='all').dropna(axis=1, how='all')


def drop_unfinished(close, now=None):
    """排程若在盤中執行，Yahoo 會把「盤中價」當成當天收盤價。這裡把尚未收盤的當日價格丟掉，避免用半天的資料做判斷。
    台股以 14:00（台北）、美股以 16:30（紐約）為界。回傳（處理後的表格, 被丟掉的說明清單）。"""
    now = now or pd.Timestamp.now(tz='Asia/Taipei')
    close, dropped = close.copy(), []
    if close.empty:
        return close, dropped
    last = close.index[-1].date()
    tw = [c for c in close.columns if c in TW_COLS]
    us = [c for c in close.columns if c in US_COLS]
    if tw and last == now.date() and (now.hour, now.minute) < (14, 0) and close.loc[close.index[-1], tw].notna().any():
        close.loc[close.index[-1], tw] = np.nan
        dropped.append(f'台股 {last} 盤中價')
    ny = now.tz_convert('America/New_York')
    if us and last == ny.date() and (ny.hour, ny.minute) < (16, 30) and close.loc[close.index[-1], us].notna().any():
        close.loc[close.index[-1], us] = np.nan
        dropped.append(f'美股 {last} 盤中價')
    return close.dropna(how='all'), dropped


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def perf_stats(px_df, names=label):
    rows = {}
    for t in px_df.columns:
        s = px_df[t].dropna()
        if len(s) < 30:
            continue
        r = s.pct_change().dropna()
        years = (s.index[-1] - s.index[0]).days / 365.25
        cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / years) - 1
        vol = r.std() * np.sqrt(TRADING_DAYS)
        rows[names(t)] = {
            '年化報酬': cagr, '年化波動': vol,
            '夏普值': cagr / vol if vol > 0 else np.nan,
            '最大回撤': (s / s.cummax() - 1).min(),
            '期間總報酬': s.iloc[-1] / s.iloc[0] - 1,
        }
    return pd.DataFrame(rows).T


BINS = [-np.inf, -0.02, -0.01, 0, 0.01, 0.02, np.inf]
BIN_NAMES = ['跌超過 2%', '跌 1~2%', '跌 0~1%', '漲 0~1%', '漲 1~2%', '漲超過 2%']


class Market:
    """把收盤價表格拆成美股、台股、匯率，並算好常用的衍生資料。"""

    def __init__(self, close_all):
        self.close_all = close_all
        self.us_cols = [t for t in US_COLS if t in close_all.columns]
        self.tw_cols = [t for t in TW_COLS if t in close_all.columns]
        self.sel_us = [t for t in US_TICKERS if t in close_all.columns]
        self.sel_tw = [t for t in TW_TICKERS if t in close_all.columns]
        self.us_close = close_all[self.us_cols].dropna(how='all')
        self.tw_close = close_all[self.tw_cols].dropna(how='all')
        self.usdtwd = close_all[FX_TICKER].dropna()
        cal = self.us_close.index.union(self.tw_close.index)
        # 每個台股交易日開盤前已知的美股收盤價（只用嚴格早於當日的資料）
        self.us_known = self.us_close.reindex(cal).ffill().shift(1).reindex(self.tw_close.index)
        self.us_on = self.us_known.pct_change(fill_method=None)
        self.tw_ret = self.tw_close.pct_change(fill_method=None)

    def is_us(self, t):
        return t in self.us_close.columns

    def normalized(self, tickers, currency):
        """多檔標的換算成同一幣別後的價格表（合併日曆、休市日沿用前值）。"""
        cal = self.us_close.index.union(self.tw_close.index)
        fx = self.usdtwd.reindex(cal.union(self.usdtwd.index)).ffill().reindex(cal)
        cols = {}
        for t in tickers:
            s = (self.us_close if self.is_us(t) else self.tw_close)[t].reindex(cal).ffill()
            if currency == '美元' and not self.is_us(t):
                s = s / fx
            if currency == '新台幣' and self.is_us(t):
                s = s * fx
            cols[t] = s
        return pd.DataFrame(cols).dropna()

    def leadlag(self, us, tw):
        pair = pd.DataFrame({'us': self.us_on[us], 'tw': self.tw_ret[tw]}).dropna()
        pair['區間'] = pd.cut(pair['us'], bins=BINS, labels=BIN_NAMES)
        g = pair.groupby('區間', observed=False)['tw']
        bucket = pd.DataFrame({
            '天數': g.size(), '台股平均報酬': g.mean(),
            '台股上漲機率': g.apply(lambda s: (s > 0).mean() if len(s) else np.nan),
        })
        return pair, bucket

    def pending_overnight(self, us):
        """最近一個台股收盤之後，美股已累積的漲跌幅；美股尚無新收盤時回傳 None。"""
        s = self.us_close[us].dropna()
        d_tw = self.tw_close.index[-1]
        known = self.us_known[us].loc[d_tw]
        if s.index[-1] < d_tw or pd.isna(known):
            return None
        return s.iloc[-1] / known - 1

    def adr_premium(self):
        prem = {}
        for adr, (tw, ratio, nm) in ADR_PAIRS.items():
            if adr not in self.us_close.columns or tw not in self.tw_close.columns:
                continue
            df = pd.concat([self.us_close[adr], self.tw_close[tw], self.usdtwd], axis=1, keys=['adr', 'tw', 'fx'])
            df['fx'] = df['fx'].ffill()
            df = df.dropna()
            prem[nm] = df['adr'] * df['fx'] / ratio / df['tw'] - 1
        return pd.DataFrame(prem)


# ───────────── 白話解讀 ─────────────

def move_word(x):
    if x >= 0.015: return '大漲'
    if x >= 0.003: return '上漲'
    if x > -0.003: return '大致持平'
    if x > -0.015: return '下跌'
    return '大跌'


def trend_word(s):
    """依股價相對季線（60 日）與年線（240 日）的位置，給一句趨勢描述。"""
    if len(s) < 240:
        return '資料不足一年，無法判斷長期趨勢'
    last, ma60, ma240 = s.iloc[-1], s.tail(60).mean(), s.tail(240).mean()
    if last >= ma240 and last >= ma60: return '長期與中期趨勢都向上'
    if last >= ma240: return '長期趨勢向上，但近期轉弱'
    if last >= ma60: return '長期趨勢偏弱，近期有反彈'
    return '長期與中期趨勢都偏弱'


def heat_word(r):
    if pd.isna(r): return '無資料'
    if r >= 70: return f'短期偏熱（RSI {r:.0f}）'
    if r <= 30: return f'短期偏冷（RSI {r:.0f}）'
    return f'中性（RSI {r:.0f}）'


def describe_stock(s, name):
    """單一標的的白話解讀（Markdown）。"""
    s = s.dropna()
    out = [f'### {name}', f'資料至 {s.index[-1]:%Y-%m-%d}，收盤 {s.iloc[-1]:,.2f}']
    if len(s) > 1:
        out[-1] += f'，單日 {s.iloc[-1] / s.iloc[-2] - 1:+.2%}'
    out.append(f'- **趨勢**：{trend_word(s)}')
    out.append(f'- **冷熱**：{heat_word(rsi(s).iloc[-1])}')
    y = s.tail(TRADING_DAYS + 1)
    if len(y) > 60:
        vol = y.pct_change().std() * np.sqrt(TRADING_DAYS)
        lvl = '低' if vol < 0.2 else ('中等' if vol < 0.35 else '高')
        out.append(f'- **近一年報酬**：{y.iloc[-1] / y.iloc[0] - 1:+.1%}')
        out.append(f'- **波動程度**：{lvl}（年化波動 {vol:.0%}，數字越大代表漲跌越劇烈）')
        dd = (y / y.cummax() - 1)
        out.append(f'- **近一年最大回撤**：{dd.min():.1%}（買在最高點最多會賠這麼多）；'
                   f'目前距離一年內高點 {dd.iloc[-1]:.1%}')
    return '\n'.join(out)


GLOSSARY = """
<details><summary>名詞說明（點擊展開）</summary>

- **年線／季線**：過去 240 個／60 個交易日的平均收盤價。股價在年線之上，代表過去一年買進的人平均是賺錢的，一般視為長期趨勢向上。
- **RSI**：0 到 100 的冷熱指標。高於 70 代表短期漲多、偏熱；低於 30 代表短期跌深、偏冷。偏熱不代表馬上會跌，只是提醒追高的風險較高。
- **費城半導體指數**：美國半導體類股的指數。台股中半導體比重很高，所以它前一晚的漲跌對台股開盤影響最直接。
- **ADR 溢價**：台積電等公司在美國掛牌的憑證，換算後比台股貴多少。比平常高，代表外資買盤比台灣投資人更積極。
- **最大回撤**：從最高點跌到最低點的最大跌幅，用來衡量「最慘會賠多少」。
- **歷史上漲機率**：過去類似情況中，台股當天收漲的比例。只是歷史統計，不是預測。

</details>
"""


def build_summary(m):
    """產生「今日摘要」的 Markdown 文字。"""
    L = []
    d_us, d_tw = m.us_close.index[-1], m.tw_close.index[-1]
    L.append(f'美股資料至 **{d_us:%Y-%m-%d}**　｜　台股資料至 **{d_tw:%Y-%m-%d}**')

    L.append('\n### 一、最近一個美股交易日')
    for t in [x for x in m.sel_us if x.startswith('^')]:
        s = m.us_close[t].dropna()
        ch = s.iloc[-1] / s.iloc[-2] - 1
        L.append(f'- {NAMES[t]}{move_word(ch)} **{ch:+.2%}**，收 {s.iloc[-1]:,.0f}')

    L.append('\n### 二、對下一個台股交易日的參考')
    if US_LEAD in m.us_close.columns and TW_TARGET in m.tw_close.columns:
        pend = m.pending_overnight(US_LEAD)
        if pend is None:
            L.append(f'台股 {d_tw:%m-%d} 收盤後，美股還沒有新的收盤資料，暫無隔夜參考。')
        else:
            _, bucket = m.leadlag(US_LEAD, TW_TARGET)
            name = pd.cut([pend], bins=BINS, labels=BIN_NAMES)[0]
            row = bucket.loc[name]
            L.append(f'台股上次收盤後，{NAMES[US_LEAD]}累計 **{pend:+.2%}**，屬於「{name}」這一組。')
            L.append(f'過去 {int(row["天數"])} 次類似情況中，{NAMES[TW_TARGET]}當天平均 '
                     f'**{row["台股平均報酬"]:+.2%}**，收漲的比例為 **{row["台股上漲機率"]:.0%}**。')
            if row['天數'] < 30:
                L.append('（這一組的歷史樣本偏少，參考價值有限。）')
            L.append('這段漲跌多半在開盤跳空時就已反映，不代表開盤後還有同樣的空間。')

    L.append('\n### 三、台股目前的位置')
    if TW_TARGET in m.tw_close.columns:
        s = m.tw_close[TW_TARGET].dropna()
        L.append(f'- {NAMES[TW_TARGET]}收 {s.iloc[-1]:,.0f}（單日 {s.iloc[-1] / s.iloc[-2] - 1:+.2%}）')
        L.append(f'- 趨勢：{trend_word(s)}')
        L.append(f'- 冷熱：{heat_word(rsi(s).iloc[-1])}')

    fx = m.usdtwd
    ch = fx.iloc[-1] / fx.iloc[-2] - 1
    word = '大致持平' if abs(ch) < 0.001 else ('新台幣貶值' if ch > 0 else '新台幣升值')
    L.append('\n### 四、匯率')
    L.append(f'1 美元兌 {fx.iloc[-1]:.2f} 新台幣，{word}（{ch:+.2%}）。'
             '新台幣貶值時，手上的美股換回台幣會變多；外資匯出時也常伴隨貶值。')

    prem = m.adr_premium()
    if '台積電' in prem.columns:
        p = prem['台積電'].dropna()
        now, avg = p.iloc[-1], p.tail(TRADING_DAYS).mean()
        cmp_ = '高於' if now > avg + 0.02 else ('低於' if now < avg - 0.02 else '接近')
        L.append('\n### 五、台積電 ADR 溢價')
        L.append(f'目前 **{now:+.1%}**，{cmp_}近一年平均（{avg:+.1%}）。')

    L.append('\n### 六、關注清單')
    L.append('| 標的 | 收盤 | 單日 | 趨勢 | 冷熱 |')
    L.append('|---|---:|---:|---|---|')
    for t in m.sel_us + m.sel_tw:
        s = m.close_all[t].dropna()
        L.append(f'| {label(t)} | {s.iloc[-1]:,.2f} | {s.iloc[-1] / s.iloc[-2] - 1:+.2%} '
                 f'| {trend_word(s)} | {heat_word(rsi(s).iloc[-1])} |')

    L.append(GLOSSARY)
    L.append('*以上為依歷史數據與固定規則自動產生的整理，僅供研究與學習，不構成投資建議。*')
    return '\n'.join(L)


# ───────────── 新聞蒐集 ─────────────

HEADERS = {'User-Agent': 'Mozilla/5.0'}
NEWS_COLS = ['time', 'market', 'source', 'ticker', 'title', 'summary', 'url']


def _to_taipei(ts):
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize('UTC')
    return ts.tz_convert('Asia/Taipei').tz_localize(None)


def fetch_cnyes(category, market, limit=30):
    import requests
    r = requests.get(f'https://api.cnyes.com/media/api/v1/newslist/category/{category}',
                     params={'limit': limit, 'page': 1}, headers=HEADERS, timeout=15)
    r.raise_for_status()
    rows = []
    for it in r.json()['items']['data']:
        codes = ','.join(str(x.get('code', '')) for x in (it.get('market') or []))
        rows.append({'time': _to_taipei(pd.Timestamp(int(it['publishAt']), unit='s', tz='UTC')),
                     'market': market, 'source': '鉅亨網', 'ticker': codes,
                     'title': it.get('title', ''), 'summary': it.get('summary') or '',
                     'url': f"https://news.cnyes.com/news/id/{it.get('newsId', '')}"})
    return rows


def fetch_twse_announcements(watch_codes=None):
    import requests
    r = requests.get('https://openapi.twse.com.tw/v1/opendata/t187ap04_L', headers=HEADERS, timeout=20)
    r.raise_for_status()
    rows = []
    for rec in r.json():
        rec = {str(k).strip(): (v.strip() if isinstance(v, str) else v) for k, v in rec.items()}
        code = rec.get('公司代號', '')
        if watch_codes and code not in watch_codes:
            continue
        try:                    # 民國年月日（如 1151113）+ 時分秒（如 173005）
            d, t = rec.get('發言日期', ''), str(rec.get('發言時間', '')).zfill(6)
            ts = pd.Timestamp(year=int(d[:-4]) + 1911, month=int(d[-4:-2]), day=int(d[-2:]),
                              hour=int(t[:2]), minute=int(t[2:4]))
        except Exception:
            ts = pd.NaT
        rows.append({'time': ts, 'market': '台股', 'source': '證交所重大訊息', 'ticker': code,
                     'title': f"{rec.get('公司名稱', '')}：{rec.get('主旨', '')}",
                     'summary': rec.get('說明', ''), 'url': 'https://mops.twse.com.tw/'})
    return rows


def fetch_yahoo_news(tickers, per_ticker=10):
    import yfinance as yf
    rows = []
    for t in tickers:
        for it in (yf.Ticker(t).news or [])[:per_ticker]:
            c = it.get('content') or it          # 新舊版 yfinance 的格式不同
            if c.get('pubDate'):
                ts = _to_taipei(c['pubDate'])
            elif c.get('providerPublishTime'):
                ts = _to_taipei(pd.Timestamp(int(c['providerPublishTime']), unit='s', tz='UTC'))
            else:
                ts = pd.NaT
            url = ((c.get('canonicalUrl') or {}).get('url')
                   or (c.get('clickThroughUrl') or {}).get('url') or c.get('link', ''))
            src = (c.get('provider') or {}).get('displayName') or c.get('publisher') or 'Yahoo Finance'
            rows.append({'time': ts, 'market': '美股', 'source': src, 'ticker': t,
                         'title': c.get('title', ''), 'summary': c.get('summary') or '', 'url': url})
    return rows


def fetch_all_news():
    """回傳（新聞表格, 各來源狀態）。任何一個來源失敗都不影響其他來源。"""
    watch = {t.split('.')[0] for t in TW_COLS if not t.startswith('^')}
    us_stocks = [t for t in US_COLS if not t.startswith('^')]
    jobs = [
        ('鉅亨網 台股', lambda: fetch_cnyes('tw_stock', '台股')),
        ('鉅亨網 美股', lambda: fetch_cnyes('us_stock', '美股')),
        ('證交所重大訊息', lambda: fetch_twse_announcements(watch)),
        ('Yahoo Finance', lambda: fetch_yahoo_news(us_stocks)),
    ]
    rows, status = [], []
    for name, job in jobs:
        try:
            got = job()
            rows += got
            status.append(f'{name}：{len(got)} 則')
        except Exception as e:
            status.append(f'{name}：失敗（{type(e).__name__}: {str(e)[:80]}）')
    df = pd.DataFrame(rows, columns=NEWS_COLS)
    df = df[df['title'].astype(str).str.strip() != '']
    df = df.drop_duplicates(subset=['source', 'title']).sort_values('time', ascending=False)
    return df.reset_index(drop=True), status


def load_news():
    if not os.path.exists(NEWS_CSV):
        return pd.DataFrame(columns=NEWS_COLS)
    df = pd.read_csv(NEWS_CSV, parse_dates=['time'], dtype={'ticker': str})
    for c in ('ticker', 'summary', 'url', 'title'):
        df[c] = df[c].fillna('')
    return df


def update_archive(new_df):
    allnews = pd.concat([load_news(), new_df], ignore_index=True)
    allnews = allnews.drop_duplicates(subset=['source', 'title']).sort_values('time', ascending=False)
    os.makedirs(DATA_DIR, exist_ok=True)
    allnews.to_csv(NEWS_CSV, index=False, encoding='utf-8-sig')
    return allnews.reset_index(drop=True)


def load_market():
    close = pd.read_csv(PRICES_CSV, index_col='Date', parse_dates=True)
    return Market(close)
