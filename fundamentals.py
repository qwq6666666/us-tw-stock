"""公司體質：成長、獲利能力、財務體質、股東回報、估值五個面向的指標、評分與同業比較。

台股資料來自證交所 OpenAPI（月營收、損益表、資產負債表、本益比／殖利率），美股來自 yfinance。
結果由 update.py 每天算好存成 data/fundamentals.json，網頁只讀檔。評分是「研究輔助的粗略尺度」，不是投資建議。
"""
import json
import os
import statistics

import core

HEADERS = {'User-Agent': 'Mozilla/5.0'}
TWSE = 'https://openapi.twse.com.tw/v1/'
FUND_JSON = os.path.join(core.DATA_DIR, 'fundamentals.json')

DIMENSIONS = ['成長', '獲利能力', '財務體質', '股東回報', '估值']

INDUSTRY = {
    '01': '水泥', '02': '食品', '03': '塑膠', '04': '紡織纖維', '05': '電機機械', '06': '電器電纜',
    '08': '玻璃陶瓷', '09': '造紙', '10': '鋼鐵', '11': '橡膠', '12': '汽車', '14': '建材營造',
    '15': '航運', '16': '觀光餐旅', '17': '金融保險', '18': '貿易百貨', '20': '其他', '21': '化學',
    '22': '生技醫療', '23': '油電燃氣', '24': '半導體', '25': '電腦及週邊設備', '26': '光電',
    '27': '通信網路', '28': '電子零組件', '29': '電子通路', '30': '資訊服務', '31': '其他電子',
    '35': '綠能環保', '36': '數位雲端', '37': '運動休閒', '38': '居家生活',
}

# 指標定義：key -> (中文名, 格式, 白話說明)。格式 pct=百分比、x=倍數、num=一般數字
METRICS = {
    'rev_yoy':   ('最新單月營收年增率', 'pct', '這個月的營收比去年同月多（少）幾 %'),
    'cum_yoy':   ('累計營收年增率', 'pct', '今年到目前的營收總和，比去年同期多（少）幾 %'),
    'earn_yoy':  ('獲利年增率', 'pct', '最近一季獲利比去年同期多（少）幾 %'),
    'gross':     ('毛利率', 'pct', '每賣 100 元，扣掉直接成本後剩幾元；越高代表產品越有定價能力'),
    'op':        ('營業利益率', 'pct', '每賣 100 元，扣掉所有營運費用後剩幾元，是本業賺錢能力'),
    'net':       ('淨利率', 'pct', '每賣 100 元，最後落袋幾元（含業外、稅）'),
    'roe':       ('股東權益報酬率 ROE', 'pct', '股東投入的錢一年能賺回幾 %；台股以累計淨利年化估算'),
    'debt':      ('負債比', 'pct', '總資產中有幾 % 是借來的；越低越穩（金融業不適用）'),
    'current':   ('流動比率', 'x', '一年內到期的負債，手上短期資產夠還幾倍；高於 1.5 倍較安心'),
    'yield':     ('現金殖利率', 'pct', '以目前股價買進，一年領到的股利約占股價幾 %'),
    'pe':        ('本益比 PE', 'x', '股價是每股盈餘的幾倍；越低代表同樣獲利買得越便宜（虧損時無意義）'),
    'fwd_pe':    ('預估本益比', 'x', '用分析師預估的未來獲利算的本益比'),
    'pb':        ('股價淨值比 PB', 'x', '股價是每股淨值的幾倍；越低越接近帳面價值'),
}

# 每個面向用哪些指標、怎麼評分：(指標, 0 分的值, 100 分的值)。0 分值 > 100 分值代表「越低越好」。
SCORING = {
    '成長':     [('rev_yoy', -0.10, 0.30), ('cum_yoy', -0.05, 0.25), ('earn_yoy', -0.20, 0.30)],
    '獲利能力': [('op', 0.0, 0.25), ('net', 0.0, 0.20), ('roe', 0.0, 0.20)],
    '財務體質': [('debt', 0.80, 0.30), ('current', 0.8, 2.0)],
    '股東回報': [('yield', 0.0, 0.04)],
    '估值':     [('pe', 35.0, 10.0), ('fwd_pe', 35.0, 10.0), ('pb', 6.0, 1.0)],
}

# 同業比較表：(指標, 是否越高越好)
PEER_METRICS = [('rev_yoy', True), ('op', True), ('net', True), ('roe', True),
                ('debt', False), ('yield', True), ('pe', False), ('pb', False)]


def _num(v):
    try:
        if v is None or (isinstance(v, str) and v.strip() in ('', '-', 'N/A')):
            return None
        x = float(str(v).replace(',', ''))
        return None if x != x else x
    except (TypeError, ValueError):
        return None


def _pct(v):
    x = _num(v)
    return x / 100 if x is not None else None


def _div(a, b):
    return a / b if a is not None and b not in (None, 0) else None


def score_one(v, lo, hi):
    if v is None:
        return None
    return max(0.0, min(1.0, (v - lo) / (hi - lo))) * 100


def score_stock(m):
    """回傳（各面向分數, 總分）。面向內缺資料的指標略過；整個面向沒資料則為 None。"""
    scores = {}
    for dim, rules in SCORING.items():
        parts = []
        for k, lo, hi in rules:
            if k == 'pe' and m.get('pe') is None and m.get('eps_negative'):
                parts.append(0.0)           # 虧損：本益比無意義，估值面給 0
                continue
            s = score_one(m.get(k), lo, hi)
            if s is not None:
                parts.append(s)
        scores[dim] = round(sum(parts) / len(parts), 1) if parts else None
    got = [v for v in scores.values() if v is not None]
    return scores, (round(sum(got) / len(got), 1) if got else None)


def grade(total):
    if total is None:
        return '資料不足'
    return '體質強' if total >= 70 else '中上' if total >= 55 else '普通' if total >= 40 else '偏弱'


def verdict(name, scores, total, note=''):
    """白話結論。"""
    have = {k: v for k, v in scores.items() if v is not None}
    if not have:
        return f'{name}：資料不足，無法評分。'
    best = max(have, key=have.get)
    worst = min(have, key=have.get)
    txt = f'{name}的體質綜合 {total:.0f} 分（{grade(total)}）。最亮眼的是「{best}」（{have[best]:.0f} 分）'
    if worst != best:
        txt += f'，相對弱的是「{worst}」（{have[worst]:.0f} 分）'
    txt += '。'
    if scores.get('估值') is not None and scores['估值'] < 35 and (scores.get('成長') or 0) >= 60:
        txt += '成長強但股價已反映不少期待，估值偏貴，若成長放緩股價較容易大幅修正。'
    elif scores.get('估值') is not None and scores['估值'] >= 70 and (scores.get('成長') or 100) < 40:
        txt += '估值便宜但成長偏弱，要留意是「便宜有原因」。'
    return txt + note


# ───────────── 台股：證交所 OpenAPI ─────────────
def _get(path):
    import requests
    r = requests.get(TWSE + path, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()


def _by_code(rows, key='公司代號'):
    return {str(r.get(key, '')).strip(): r for r in rows}


def fetch_tw_all():
    """下載全部上市公司的最新財報資料，回傳 (每家公司的指標 dict, 期間說明)。"""
    rev = _by_code(_get('opendata/t187ap05_L'))
    inc = _by_code(_get('opendata/t187ap06_L_ci'))
    bal = _by_code(_get('opendata/t187ap07_L_ci'))
    val = _by_code(_get('exchangeReport/BWIBBU_ALL'), 'Code')
    basic = _by_code(_get('opendata/t187ap03_L'))
    out, period = {}, ''
    for code, i in inc.items():
        b = bal.get(code, {})
        r = rev.get(code, {})
        v = val.get(code, {})
        q = int(_num(i.get('季別')) or 0)
        sales = _num(i.get('營業收入'))
        ni_parent = _num(i.get('淨利（淨損）歸屬於母公司業主'))
        if ni_parent is None:
            ni_parent = _num(i.get('本期淨利（淨損）'))
        equity = _num(b.get('歸屬於母公司業主之權益合計'))
        pe = _num(v.get('PEratio'))
        m = {
            'rev_yoy': _pct(r.get('營業收入-去年同月增減(%)')),
            'cum_yoy': _pct(r.get('累計營業收入-前期比較增減(%)')),
            'gross': _div(_num(i.get('營業毛利（毛損）淨額')), sales),
            'op': _div(_num(i.get('營業利益（損失）')), sales),
            'net': _div(_num(i.get('本期淨利（淨損）')), sales),
            'roe': _div(ni_parent * 4 / q if ni_parent is not None and q else None, equity),
            'debt': _div(_num(b.get('負債總計')), _num(b.get('資產總計'))),
            'current': _div(_num(b.get('流動資產')), _num(b.get('流動負債'))),
            'yield': _pct(v.get('DividendYield')),
            'pe': pe,
            'pb': _num(v.get('PBratio')),
            'eps': _num(i.get('基本每股盈餘（元）')),
        }
        m['eps_negative'] = m['eps'] is not None and m['eps'] < 0
        ind = str(basic.get(code, {}).get('產業別', '')).strip()
        m['industry'] = INDUSTRY.get(ind, ind)
        m['name'] = str(i.get('公司名稱', '')).strip()
        out[code] = m
        period = f"{int(_num(i.get('年度')) or 0) + 1911} 年第 {q} 季財報（損益為年初累計）"
        ym = str(r.get('資料年月', '')).strip()
        if ym:
            period += f"；月營收 {int(ym[:-2]) + 1911}/{ym[-2:]}"
    return out, period


def _pctile_rank(values, x, higher_better):
    """x 在 values 中的排名百分位（100 = 同業最好）。"""
    vals = [v for v in values if v is not None]
    if x is None or len(vals) < 5:
        return None
    better = sum(1 for v in vals if (v < x if higher_better else v > x))
    return round(100 * better / (len(vals) - 1 if len(vals) > 1 else 1))


def peer_rows(code, universe, group_keys, label_peer):
    """某檔股票與同業的比較表。universe: {代號: 指標}；group_keys: 同業代號清單。"""
    me = universe[code]
    rows = []
    for k, hb in PEER_METRICS:
        vals = [universe[c].get(k) for c in group_keys if c != code and universe[c].get(k) is not None]
        if k == 'pe':
            vals = [v for v in vals if v > 0]
        med = statistics.median(vals) if vals else None
        allv = vals + ([me.get(k)] if me.get(k) is not None else [])
        rows.append({'key': k, 'mine': me.get(k), 'median': med,
                     'rank': _pctile_rank(allv, me.get(k), hb), 'n': len(allv)})
    return rows


# ───────────── 美股：yfinance ─────────────
def fetch_us(ticker):
    import yfinance as yf
    t = yf.Ticker(ticker)
    i = t.info or {}
    dy = _num(i.get('dividendYield'))
    debt_ratio = None             # 與台股同一定義：總負債 / 總資產（取自最新一季資產負債表）
    try:
        bs = t.quarterly_balance_sheet
        col = bs.columns[0]
        debt_ratio = _div(_num(bs.at['Total Liabilities Net Minority Interest', col]), _num(bs.at['Total Assets', col]))
    except Exception:
        pass
    same_ccy = i.get('financialCurrency') in (None, i.get('currency'))
    pe = _num(i.get('trailingPE'))
    eps = _num(i.get('trailingEps'))
    m = {
        'rev_yoy': _num(i.get('revenueGrowth')),            # 最近一季營收年增率
        'cum_yoy': None,
        'earn_yoy': _num(i.get('earningsGrowth')),
        'gross': _num(i.get('grossMargins')),
        'op': _num(i.get('operatingMargins')),
        'net': _num(i.get('profitMargins')),
        'roe': _num(i.get('returnOnEquity')),
        'debt': debt_ratio,
        'current': _num(i.get('currentRatio')),
        'yield': (dy / 100) if dy is not None else None,    # yfinance 1.x 以百分比表示
        'pe': pe,
        'fwd_pe': _num(i.get('forwardPE')),
        'pb': _num(i.get('priceToBook')) if same_ccy else None,   # ADR 的股價與財報幣別不同時，PB 不可信
        'eps': eps,
        'eps_negative': eps is not None and eps < 0,
        'industry': i.get('industry') or i.get('sector') or '',
        'name': core.NAMES.get(ticker, i.get('shortName', ticker)),
    }
    return m


# ───────────── 整合 ─────────────
def _pack(code, m, universe, group, note=''):
    scores, total = score_stock(m)
    clean = {k: m.get(k) for k in METRICS if m.get(k) is not None}
    return {'name': m['name'], 'industry': m['industry'], 'metrics': clean, 'scores': scores, 'total': total,
            'grade': grade(total), 'verdict': verdict(m['name'], scores, total, note),
            'peers': peer_rows(code, universe, group, ''), 'peer_group': len(group)}


def build_all():
    """算出關注清單內所有個股的體質；任一市場失敗不影響另一個。回傳可存成 JSON 的 dict。"""
    from datetime import datetime, timezone, timedelta
    result = {'updated_at': datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M'),
              'tw': {}, 'us': {}, 'status': []}

    try:
        uni, period = fetch_tw_all()
        watch = [t.split('.')[0] for t in core.TW_COLS if not t.startswith('^') and t.endswith('.TW')]
        for code in watch:
            if code not in uni:
                continue               # ETF（如 0050）沒有公司財報
            ind = uni[code]['industry']
            group = [c for c, v in uni.items() if v['industry'] == ind]
            note = ''
            if ind == '金融保險':
                note = '（金融業的負債比、流動比不適用，評分僅供參考）'
            result['tw'][code] = _pack(code, uni[code], uni, group, note)
            result['tw'][code]['peer_label'] = f'上市「{ind}」同業 {len(group)} 家'
        result['tw_period'] = period
        result['status'].append(f'台股：{len(result["tw"])} 檔（全上市 {len(uni)} 家作同業比較）')
    except Exception as e:
        result['status'].append(f'台股失敗（{type(e).__name__}: {str(e)[:80]}）')

    us_tickers = [t for t in core.US_COLS if not t.startswith('^')]
    us_m = {}
    for t in us_tickers:
        try:
            us_m[t] = fetch_us(t)
        except Exception as e:
            result['status'].append(f'美股 {t} 失敗（{type(e).__name__}: {str(e)[:60]}）')
    for t, m in us_m.items():
        note = '（ADR：財報以台幣等外幣計，比率不受影響，但股價與財報幣別不同時不計股價淨值比）' if t in core.ADR_PAIRS else ''
        result['us'][t] = _pack(t, m, us_m, list(us_m), note)
        result['us'][t]['peer_label'] = f'關注清單內美股 {len(us_m)} 檔（樣本少，僅供參考）'
    result['status'].append(f'美股：{len(result["us"])} 檔')
    return result


def save(result):
    os.makedirs(core.DATA_DIR, exist_ok=True)
    with open(FUND_JSON, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=1)


def load():
    if not os.path.exists(FUND_JSON):
        return None
    with open(FUND_JSON, encoding='utf-8') as f:
        return json.load(f)


if __name__ == '__main__':
    res = build_all()
    save(res)
    print('\n'.join(res['status']))
