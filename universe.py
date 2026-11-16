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
# ───────────── 第二批擴充（降低存活者偏誤） ─────────────
# 原名單只有「現在的」大型股。這一批刻意納入 2020~2021 年是大型或熱門股、之後表現不佳的公司（例如波音、英特爾同業、派拉蒙、Peloton、
# 宏達電、大同、友達同業等），以及各產業的中型股，讓強者與弱者都在池子裡。
# 仍然做不到的：已下市、被併購或改名而 Yahoo Finance 查不到的公司（例如 Twitter、動視暴雪、Walgreens、美國鋼鐵、Anthem 舊代號），
# 它們無法納入，所以仍有存活者偏誤，只是比原名單小。這批代號在本機驗證時下載不到的清單見 UNAVAILABLE。
TW_EXTRA = {
    '2105.TW': '正新', '2801.TW': '彰銀', '1402.TW': '遠東新', '2227.TW': '裕日車',
    '2356.TW': '英業達', '2449.TW': '京元電子', '2492.TW': '華新科', '3406.TW': '玉晶光',
    '3702.TW': '大聯大', '4958.TW': '臻鼎-KY', '5876.TW': '上海商銀', '6415.TW': '矽力*-KY',
    '8046.TW': '南電', '9904.TW': '寶成', '9921.TW': '巨大', '2371.TW': '大同',
    '2324.TW': '仁寶', '2347.TW': '聯強', '2385.TW': '群光', '2618.TW': '長榮航',
    '2610.TW': '華航', '2633.TW': '台灣高鐵', '1590.TW': '亞德客-KY', '2049.TW': '上銀',
    '1476.TW': '儒鴻', '6770.TW': '力積電', '2344.TW': '華邦電', '2337.TW': '旺宏',
    '3035.TW': '智原', '3443.TW': '創意', '5269.TW': '祥碩', '6488.TWO': '環球晶',
    '8299.TWO': '群聯', '2383.TW': '台光電', '2368.TW': '金像電', '3044.TW': '健鼎',
    '2313.TW': '華通', '3653.TW': '健策', '6239.TW': '力成', '2404.TW': '漢唐',
    '9945.TW': '潤泰新', '2542.TW': '興富發', '2548.TW': '華固', '1795.TW': '美時',
    '6446.TW': '藥華藥', '1102.TW': '亞泥', '1210.TW': '大成', '1227.TW': '佳格',
    '1229.TW': '聯華', '1319.TW': '東陽', '1434.TW': '福懋', '1504.TW': '東元',
    '1513.TW': '中興電', '1519.TW': '華城', '1605.TW': '華新', '1722.TW': '台肥',
    '2006.TW': '東鋼', '2015.TW': '豐興', '2059.TW': '川湖', '2101.TW': '南港',
    '2201.TW': '裕隆', '2204.TW': '中華', '2231.TW': '為升', '2312.TW': '金寶',
    '2352.TW': '佳世達', '2355.TW': '敬鵬', '2360.TW': '致茂', '2374.TW': '佳能',
    '2481.TW': '強茂', '2498.TW': '宏達電', '2504.TW': '國產', '2515.TW': '中工',
    '2520.TW': '冠德', '2524.TW': '京城', '2530.TW': '華建', '3019.TW': '亞光',
    '3036.TW': '文曄', '3105.TWO': '穩懋', '3189.TW': '景碩', '3293.TWO': '鈊象',
    '3376.TW': '新日興', '3515.TW': '華擎', '3552.TWO': '同致', '3596.TW': '智易',
    '3665.TW': '貿聯-KY', '4915.TW': '致伸', '4919.TW': '新唐', '5388.TW': '中磊',
    '6121.TWO': '新普', '6176.TW': '瑞儀', '6214.TW': '精誠', '6409.TW': '旭隼',
    '6531.TW': '愛普*', '6533.TW': '晶心科', '6547.TWO': '高端疫苗', '8016.TW': '矽創',
    '8069.TWO': '元太', '8454.TW': '富邦媒', '6592.TW': '和潤企業', '5522.TW': '遠雄',
    '2905.TW': '三商', '2915.TW': '潤泰全', '2929.TW': '淳紳', '8464.TW': '億豐',
    '9914.TW': '美利達', '9917.TW': '中保科', '9941.TW': '裕融',
}
US_EXTRA = {
    'BA': '波音', 'GE': '奇異', 'PFE': '輝瑞', 'MRK': '默克',
    'ABBV': '艾伯維', 'T': 'AT&T', 'VZ': '威瑞森', 'CMCSA': '康卡斯特',
    'CVX': '雪佛龍', 'WFC': '富國銀行', 'C': '花旗', 'GS': '高盛',
    'MS': '摩根士丹利', 'IBM': 'IBM', 'MMM': '3M', 'CAT': '開拓重工',
    'DE': '迪爾', 'HON': '漢威聯合', 'UPS': 'UPS', 'FDX': '聯邦快遞',
    'LMT': '洛克希德馬丁', 'RTX': '雷神技術', 'NKE': '耐吉', 'MCD': '麥當勞',
    'SBUX': '星巴克', 'TGT': '塔吉特', 'LOW': '勞氏', 'F': '福特',
    'GM': '通用汽車', 'CVS': 'CVS', 'KHC': '卡夫亨氏', 'MDLZ': '億滋',
    'PM': '菲利普莫里斯', 'MO': '奧馳亞', 'GILD': '吉利德', 'BMY': '必治妥',
    'AMGN': '安進', 'ABT': '亞培', 'TMO': '賽默飛', 'DHR': '丹納赫',
    'MDT': '美敦力', 'ADBE': 'Adobe', 'PYPL': 'PayPal', 'INTU': '財捷',
    'NOW': 'ServiceNow', 'UBER': 'Uber', 'LYFT': 'Lyft', 'PARA': '派拉蒙',
    'VFC': 'VF', 'LUMN': 'Lumen', 'XRX': '全錄', 'MRNA': 'Moderna',
    'ZM': 'Zoom', 'PTON': 'Peloton', 'SNAP': 'Snap', 'HPQ': '惠普',
    'DELL': '戴爾', 'KSS': '柯爾百貨', 'M': '梅西百貨', 'NWL': 'Newell',
    'AIG': 'AIG', 'MET': '大都會人壽', 'PRU': '保德信', 'USB': '美國合眾銀行',
    'PNC': 'PNC', 'SCHW': '嘉信理財', 'BLK': '貝萊德', 'AXP': '美國運通',
    'COF': '第一資本', 'SPGI': '標普全球', 'LIN': '林德', 'APD': '空氣化工',
    'DUK': '杜克能源', 'SO': '南方電力', 'NEE': 'NextEra', 'D': '道明尼能源',
    'EXC': 'Exelon', 'COP': '康菲', 'OXY': '西方石油', 'SLB': '斯倫貝謝',
    'KMI': 'Kinder Morgan', 'PSX': 'Phillips 66', 'MPC': 'Marathon Petroleum', 'DOW': '陶氏',
    'DD': '杜邦', 'LYB': '利安德巴塞爾', 'NEM': '紐蒙特', 'FCX': '自由港麥克莫蘭',
    'MCK': '麥克森', 'CI': '信諾', 'ELV': 'Elevance', 'HUM': '哈門那',
    'ISRG': '直覺外科', 'SYK': '史賽克', 'BDX': 'BD', 'ZTS': '碩騰',
    'REGN': '再生元', 'VRTX': '福泰', 'BIIB': '百健', 'CL': '高露潔',
    'KMB': '金百利克拉克', 'GIS': '通用磨坊', 'HSY': '好時', 'EL': '雅詩蘭黛',
    'LVS': '拉斯維加斯金沙', 'MAR': '萬豪', 'HLT': '希爾頓', 'BKNG': 'Booking',
    'EXPE': 'Expedia', 'ABNB': 'Airbnb', 'ETSY': 'Etsy', 'EBAY': 'eBay',
    'DASH': 'DoorDash', 'SHOP': 'Shopify', 'CRWD': 'CrowdStrike', 'PANW': 'Palo Alto',
    'SNOW': 'Snowflake', 'WDAY': 'Workday', 'TEAM': 'Atlassian', 'ADSK': 'Autodesk',
    'ANET': 'Arista', 'LRCX': '科林研發', 'KLAC': '科磊', 'ADI': '亞德諾',
    'MCHP': '微芯', 'NXPI': '恩智浦', 'ON': '安森美', 'WDC': '威騰電子',
    'STX': '希捷', 'HPE': '慧與', 'CTSH': 'Cognizant', 'ACN': '埃森哲',
    'FIS': 'FIS', 'FISV': 'Fiserv', 'GPN': 'Global Payments', 'TTWO': 'Take-Two',
    'SPOT': 'Spotify', 'LMND': 'Lemonade', 'OPEN': 'Opendoor', 'CCL': '嘉年華郵輪',
    'RCL': '皇家加勒比', 'NCLH': '挪威郵輪', 'AAL': '美國航空', 'DAL': '達美航空',
    'UAL': '聯合航空', 'LUV': '西南航空', 'CLF': 'Cleveland-Cliffs', 'AA': '美鋁',
}
UNAVAILABLE = ['TWTR', 'ATVI', 'X', 'ANTM', 'K', 'WBA', 'SQ', '2823.TW', '2888.TW']     # 2021 年是大型股、現在查不到（下市／併購／改代號）
TW_CORE, US_CORE = dict(TW_UNIVERSE), dict(US_UNIVERSE)      # 原名單（現在的大型股），用來和擴充後的名單比較偏誤
TW_UNIVERSE.update(TW_EXTRA)
US_UNIVERSE.update(US_EXTRA)

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


def _clean(data):
    """依收盤價的斷點（見 core.clean_breaks）把四個欄位同步截斷，並記下被截斷的代號。"""
    close, cut = core.clean_breaks(data['close'])
    out = {k: d.copy() for k, d in data.items()}
    for c, day in cut:
        for d in out.values():
            if c in d.columns:
                d.loc[d.index < pd.Timestamp(day), c] = float('nan')
    global LAST_CUTS
    LAST_CUTS = cut
    return {k: d.dropna(how='all') for k, d in out.items()}


LAST_CUTS = []


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
    out = _clean(out)
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
