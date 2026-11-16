"""擴充標的範圍（台股大型股約 50 檔、美股大型股約 30 檔）與成交量、高低價的儲存。

重要限制（存活者偏誤）：名單是「現在的」大型股，固定不變。拿它回測過去，會漏掉曾經是大型股、後來掉出去的公司，
也一定包含後來才變大的公司，因此會讓「強者恆強」看起來比真實情況更有效。使用這份名單做的任何歷史驗證都要這樣解讀。
名單以台灣 50 成分股與美國大型股為參考，不是即時成分；個別代號若下載不到或歷史不足，自動略過。

資料存放：每個欄位（收盤、最高、最低、成交量）一個寬表 CSV，欄 = 代號、列 = 日期。價格已還原權息與分割。
  data/uni_close.csv  data/uni_high.csv  data/uni_low.csv  data/uni_volume.csv
"""
import os

import pandas as pd

import core

FIELDS = {'close': 'Close', 'high': 'High', 'low': 'Low', 'volume': 'Volume'}
PATHS = {k: os.path.join(core.DATA_DIR, f'uni_{k}.csv') for k in FIELDS}

TW_UNIVERSE = {
    '2330.TW': '台積電', '2317.TW': '鴻海', '2454.TW': '聯發科', '2308.TW': '台達電', '2382.TW': '廣達', '2412.TW': '中華電',
    '2881.TW': '富邦金', '2882.TW': '國泰金', '2891.TW': '中信金', '2886.TW': '兆豐金', '2884.TW': '玉山金', '2885.TW': '元大金',
    '2892.TW': '第一金', '2880.TW': '華南金', '2883.TW': '開發金', '2887.TW': '台新金', '2890.TW': '永豐金', '5880.TW': '合庫金',
    '5871.TW': '中租-KY', '1301.TW': '台塑', '1303.TW': '南亞', '1326.TW': '台化', '6505.TW': '台塑化', '2002.TW': '中鋼',
    '1216.TW': '統一', '2912.TW': '統一超', '3711.TW': '日月光投控', '2303.TW': '聯電', '3008.TW': '大立光', '2357.TW': '華碩',
    '2379.TW': '瑞昱', '2395.TW': '研華', '3034.TW': '聯詠', '3037.TW': '欣興', '3045.TW': '台灣大', '4904.TW': '遠傳',
    '4938.TW': '和碩', '2327.TW': '國巨', '2345.TW': '智邦', '3231.TW': '緯創', '2603.TW': '長榮', '2609.TW': '陽明',
    '2615.TW': '萬海', '2301.TW': '光寶科', '6669.TW': '緯穎', '3661.TW': '世芯-KY', '3017.TW': '奇鋐', '2376.TW': '技嘉',
    '2377.TW': '微星', '2474.TW': '可成', '1101.TW': '台泥', '2207.TW': '和泰車', '9910.TW': '豐泰', '2408.TW': '南亞科',
    '2409.TW': '友達', '3481.TW': '群創', '2353.TW': '宏碁', '0050.TW': '元大台灣50',
}
US_UNIVERSE = {
    'AAPL': '蘋果', 'MSFT': '微軟', 'NVDA': '輝達', 'AMZN': '亞馬遜', 'GOOGL': 'Alphabet', 'META': 'Meta', 'TSLA': '特斯拉',
    'AVGO': '博通', 'BRK-B': '波克夏', 'JPM': '摩根大通', 'V': 'Visa', 'MA': '萬事達卡', 'UNH': '聯合健康', 'XOM': '埃克森美孚',
    'JNJ': '嬌生', 'WMT': '沃爾瑪', 'PG': '寶僑', 'HD': '家得寶', 'COST': '好市多', 'LLY': '禮來', 'ORCL': '甲骨文',
    'AMD': 'AMD', 'NFLX': 'Netflix', 'KO': '可口可樂', 'PEP': '百事', 'CRM': 'Salesforce', 'CSCO': '思科', 'INTC': '英特爾',
    'QCOM': '高通', 'MU': '美光', 'TXN': '德州儀器', 'AMAT': '應用材料', 'DIS': '迪士尼', 'BAC': '美國銀行',
    'TSM': '台積電 ADR', 'UMC': '聯電 ADR', 'ASX': '日月光 ADR', 'CHT': '中華電信 ADR',
}
INDEXES = ['^TWII', '^GSPC', '^IXIC', '^SOX']
MIN_DAYS = 400


def tickers():
    """要下載的全部代號：擴充名單 + 原本的關注清單（含指數、匯率，讓它們也有高低價與成交量）。"""
    out = list(TW_UNIVERSE) + list(US_UNIVERSE) + core.TW_COLS + core.US_COLS
    return list(dict.fromkeys(out))


def names():
    return {**TW_UNIVERSE, **US_UNIVERSE, **{t: core.NAMES.get(t, t) for t in core.TW_COLS + core.US_COLS}}


def is_tw(t):
    return t.endswith('.TW') or t.endswith('.TWO') or t == '^TWII'


def download(tickers_, start=core.START):
    """下載還原後的收盤、最高、最低、成交量。回傳 {欄位: 寬表}；沒有資料的代號會被略過。"""
    import yfinance as yf
    def grab(tk):
        raw = yf.download(list(tk), start=start, auto_adjust=True, progress=False, group_by='column', threads=len(tk) > 1)
        res = {}
        for k, col in FIELDS.items():
            d = raw[col].copy()
            if isinstance(d, pd.Series):
                d = d.to_frame(list(tk)[0])
            d.index = pd.to_datetime(d.index).tz_localize(None)
            d.index.name = 'Date'
            res[k] = d.dropna(how='all').dropna(axis=1, how='all')
        return res

    out = grab(tickers_)
    for _ in range(2):          # 多執行緒下載偶爾會因本機快取資料庫被鎖住而漏抓幾檔：逐一補抓
        missing = [t for t in tickers_ if t not in out['close'].columns]
        if not missing:
            break
        for t in missing:
            try:
                one = grab([t])
            except Exception:
                continue
            for k in out:
                if t in one[k].columns:
                    out[k] = out[k].join(one[k][[t]], how='outer')
    return out


def drop_unfinished(data, now=None):
    """丟掉尚未收盤的當日資料（與 core.drop_unfinished 同一規則，但依代號判斷市場，涵蓋整個擴充名單）。"""
    now = now or pd.Timestamp.now(tz='Asia/Taipei')
    ny = now.tz_convert('America/New_York')
    out, dropped = {k: d.copy() for k, d in data.items()}, []
    last = max(d.index[-1] for d in data.values())
    for market, ok in (('台股', lambda c: is_tw(c)), ('美股', lambda c: not is_tw(c))):
        day = now if market == '台股' else ny
        early = (day.hour, day.minute) < ((14, 0) if market == '台股' else (16, 30))
        if last.date() == day.date() and early:
            for k, d in out.items():
                cols = [c for c in d.columns if ok(c)]
                if cols:
                    d.loc[d.index == last, cols] = float('nan')
            dropped.append(f'{market} {last.date()} 盤中資料')
    return {k: d.dropna(how='all') for k, d in out.items()}, dropped


def save(data):
    os.makedirs(core.DATA_DIR, exist_ok=True)
    for k, d in data.items():
        d.round(6).to_csv(PATHS[k], encoding='utf-8-sig')


def load():
    """回傳 {欄位: 寬表}；檔案不存在時回傳 None。"""
    if not all(os.path.exists(p) for p in PATHS.values()):
        return None
    return {k: pd.read_csv(p, index_col='Date', parse_dates=True) for k, p in PATHS.items()}


def usable(close, universe):
    """從名單中挑出歷史夠長的代號（至少 MIN_DAYS 個交易日）。"""
    return [t for t in universe if t in close.columns and close[t].notna().sum() >= MIN_DAYS]
