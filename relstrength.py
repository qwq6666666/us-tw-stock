"""相對強弱：在一群股票裡，過去表現相對強的，之後一個月是否也相對強？（橫斷面動能）

為什麼值得測：預測「會不會漲」很難，但預測「誰比誰強」在學術研究上有長期的證據（動能效應）。這裡用本專案的資料檢驗一次。

【事先固定的規則（看結果前寫死，不調參數）】
  個股池：台股 = 擴充名單中的個股（不含 ETF）；美股 = 擴充名單中的個股（不含 ADR，避免和台股重複計算）。各市場分開排名。
  主要訊號：12-1 動能 = 前 12 個月報酬，但跳過最近 1 個月（close[t-21] / close[t-252] - 1），這是文獻中最標準的定義。
  做法：每個月最後一個交易日排名；答案 = 下一個月底的報酬。只看「那天已知」的資料。
  判準：每月的排名相關係數（Spearman IC，分數高低與之後報酬高低的一致程度）平均大於 0，
        且以月為單位的區塊拔靴法 95% 信賴區間下界 > 0，才算「有證據」。
  參考訊號（只描述、不影響判定，因為同時看多個容易碰巧有一個好看）：6-1 動能、波動調整後的 12-1 動能、1 個月反轉。

【最重要的限制：存活者偏誤】
  名單是「現在的」大型股。現在是大型股的公司，過去多半表現不錯，所以回測會高估「強者恆強」。
  下面會附上「名單等權重報酬 vs 大盤指數」來粗估這個偏誤有多大。因此，沒有證據 → 可信；有證據 → 要打折看。
"""
import json
import os

import numpy as np
import pandas as pd

import core
import trend
import universe as U

REL_JSON = os.path.join(core.DATA_DIR, 'relstrength.json')
PRIMARY = 'mom12_1'
SIGNALS = {'mom12_1': '12-1 動能（主要）', 'mom6_1': '6-1 動能', 'vamom': '波動調整後 12-1 動能', 'rev1': '1 個月反轉'}
MIN_STOCKS = 20
TOP_FRAC = 0.2
N_BOOT = 2000
BLOCK = 3
ADRS = {'TSM', 'UMC', 'ASX', 'CHT'}


def pool(close, market, core_only=False):
    if core_only:
        uni = U.TW_CORE if market == 'tw' else U.US_CORE
    else:
        uni = U.TW_UNIVERSE if market == 'tw' else U.US_UNIVERSE
    ex = {'0050.TW'} | (ADRS if market == 'us' else set())
    return [t for t in U.usable(close, uni) if t not in ex]


def signals(close):
    """每個交易日的各訊號（只用到當日為止的收盤價）。"""
    r = np.log(close).diff()
    vol = r.rolling(60).std()
    mom12 = close.shift(21) / close.shift(252) - 1
    return {'mom12_1': mom12, 'mom6_1': close.shift(21) / close.shift(126) - 1,
            'vamom': mom12 / vol.where(vol > 0), 'rev1': -(close / close.shift(21) - 1)}


def month_ends(idx):
    """每個完整月份的最後一個交易日。資料最後一個（可能尚未結束的）月份不算。"""
    s = pd.Series(idx, index=idx)
    ends = s.groupby([idx.year, idx.month]).last()
    return pd.DatetimeIndex(ends.values)[:-1]


def block_ci(x, block=BLOCK, n_boot=N_BOOT, seed=0):
    x = np.asarray(x, float)
    n = len(x)
    if n < block * 4:
        return None, None
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    means = np.array([np.concatenate([x[s:s + block] for s in rng.integers(0, n - block + 1, size=k)])[:n].mean() for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate(sig, close, ends):
    """單一訊號的逐月 IC、前 20%／後 20% 報酬、換手率，以及彙總。"""
    rows, prev_top = [], None
    for a, b in zip(ends[:-1], ends[1:]):
        S, F = sig.loc[a], close.loc[b] / close.loc[a] - 1
        ok = S.notna() & F.notna()
        if ok.sum() < MIN_STOCKS:
            continue
        S, F = S[ok], F[ok]
        k = max(int(round(len(S) * TOP_FRAC)), 3)
        order = S.sort_values(ascending=False).index
        top, bot = order[:k], order[-k:]
        ic = float(S.rank().corr(F.rank()))
        row = {'date': b, 'ic': ic, 'top': float(F[top].mean()), 'bottom': float(F[bot].mean()), 'all': float(F.mean()), 'n': int(len(S))}
        row['turnover'] = None if prev_top is None else 1 - len(set(top) & set(prev_top)) / k
        prev_top = list(top)
        rows.append(row)
    if not rows:
        return None
    d = pd.DataFrame(rows).set_index('date')
    spread, excess = d['top'] - d['bottom'], d['top'] - d['all']
    ic_lo, ic_hi = block_ci(d['ic'].values)
    sp_lo, sp_hi = block_ci(spread.values)
    ex_lo, ex_hi = block_ci(excess.values)
    res = {'months': int(len(d)), 'from': d.index[0].strftime('%Y-%m'), 'to': d.index[-1].strftime('%Y-%m'),
           'ic_mean': float(d['ic'].mean()), 'ic_ci': [ic_lo, ic_hi], 'ic_pos_rate': float((d['ic'] > 0).mean()),
           'spread_mean': float(spread.mean()), 'spread_ci': [sp_lo, sp_hi],
           'excess_mean': float(excess.mean()), 'excess_ci': [ex_lo, ex_hi], 'excess_pos_rate': float((excess > 0).mean()),
           'top_mean': float(d['top'].mean()), 'bottom_mean': float(d['bottom'].mean()), 'all_mean': float(d['all'].mean()),
           'turnover': float(d['turnover'].mean()), 'avg_stocks': float(d['n'].mean()),
           'has_evidence': bool(ic_lo is not None and ic_lo > 0)}
    return res, d


def universe_vs_index(close, index_close):
    """名單等權重（每日再平衡）與大盤指數的年化報酬，用來粗估存活者偏誤。"""
    r = close.pct_change(fill_method=None).mean(axis=1).dropna()
    r = r[r.index >= close.index[252]]
    ir = index_close.dropna().pct_change().reindex(r.index).dropna()
    r = r.reindex(ir.index)
    ann = lambda x: float((1 + x).prod() ** (core.TRADING_DAYS / len(x)) - 1)
    return {'universe': ann(r), 'index': ann(ir), 'gap': ann(r) - ann(ir)}


def current_table(close, volume, market, names):
    """目前（最新一天）的排名。"""
    sig = signals(close)
    last = close.index[-1]
    S = sig[PRIMARY].loc[last]
    rank = S.rank(ascending=False)
    n = int(S.notna().sum())
    rows = []
    for t in close.columns:
        if S.get(t) != S.get(t):
            continue
        s = close[t].dropna()
        z = trend.strength(s)
        v = volume[t].dropna() if volume is not None and t in volume.columns else None
        rows.append({'ticker': t, 'name': names.get(t, t), 'score': float(S[t]), 'rank': int(rank[t]), 'n': n,
                     'quintile': int(min(5, np.ceil(rank[t] / (n / 5)))),
                     'ret1m': float(s.iloc[-1] / s.iloc[-22] - 1) if len(s) > 22 else None,
                     'ret3m': float(s.iloc[-1] / s.iloc[-64] - 1) if len(s) > 64 else None,
                     'z60': float(z.iloc[-1]) if z.iloc[-1] == z.iloc[-1] else None,
                     'vol_ratio': float(v.tail(20).mean() / v.tail(60).mean()) if v is not None and len(v) > 60 and v.tail(60).mean() > 0 else None})
    rows.sort(key=lambda r: r['rank'])
    return {'as_of': last.strftime('%Y-%m-%d'), 'rows': rows}


def build_all(data=None):
    data = data or U.load()
    if data is None:
        return None
    names = U.names()
    res = {'updated_at': pd.Timestamp.now(tz='Asia/Taipei').strftime('%Y-%m-%d %H:%M'), 'primary': PRIMARY, 'signals': SIGNALS,
           'top_frac': TOP_FRAC, 'markets': {}, 'status': []}
    for market, mname, idx_t in (('tw', '台股', '^TWII'), ('us', '美股', '^GSPC')):
        cols = pool(data['close'], market)
        if len(cols) < MIN_STOCKS:
            continue
        close = data['close'][cols].dropna(how='all')
        ends = month_ends(close.index)
        sigs = signals(close)
        out = {'n_stocks': len(cols), 'results': {}, 'curve': None}
        for k in SIGNALS:
            ev = evaluate(sigs[k], close, ends)
            if ev:
                out['results'][k] = ev[0]
                if k == PRIMARY:
                    d = ev[1]
                    cum = (1 + d[['top', 'bottom', 'all']]).cumprod()
                    out['curve'] = {'dates': [x.strftime('%Y-%m-%d') for x in d.index],
                                    'top': [round(float(x), 4) for x in cum['top']], 'bottom': [round(float(x), 4) for x in cum['bottom']],
                                    'all': [round(float(x), 4) for x in cum['all']]}
        if idx_t in data['close'].columns:
            out['bias'] = universe_vs_index(close, data['close'][idx_t])
        # 與原名單（現在的大型股）比較：擴充後偏誤是否變小？（描述性，兩個池子的主要訊號與等權重報酬）
        ccols = pool(data['close'], market, core_only=True)
        if len(ccols) >= MIN_STOCKS and len(ccols) < len(cols):
            cclose = data['close'][ccols].dropna(how='all')
            cev = evaluate(signals(cclose)[PRIMARY], cclose, month_ends(cclose.index))
            out['core_pool'] = {'n_stocks': len(ccols), 'result': cev[0] if cev else None,
                                'bias': universe_vs_index(cclose, data['close'][idx_t]) if idx_t in data['close'].columns else None}
        out['unavailable'] = U.UNAVAILABLE
        out['current'] = current_table(close, data.get('volume'), market, names)
        res['markets'][market] = out
        r = out['results'].get(PRIMARY)
        if r:
            b, cb = out.get('bias'), (out.get('core_pool') or {}).get('bias')
            gap = f"；名單等權重比大盤年化高 {b['gap']:+.1%}" + (f"（原名單 {cb['gap']:+.1%}）" if cb else '') if b else ''
            res['status'].append(f"{mname} {len(cols)} 檔：12-1 動能 IC {r['ic_mean']:+.3f}，" + ('有證據' if r['has_evidence'] else '沒有證據') + gap)
    return res


def save(res):
    os.makedirs(core.DATA_DIR, exist_ok=True)
    with open(REL_JSON, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


def load():
    if not os.path.exists(REL_JSON):
        return None
    with open(REL_JSON, encoding='utf-8') as f:
        return json.load(f)


if __name__ == '__main__':
    r = build_all()
    save(r)
    print('\n'.join(r['status']))
