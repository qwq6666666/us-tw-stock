"""趨勢判斷的實測：不預測漲跌，而是檢驗「跟著趨勢走」的規則實際表現，並提供波動調整後的趨勢強度。

一、趨勢規則回測（規則與判準在看結果前固定，不調參數）
  規則都是常見的經典設定，不是從這份資料挑出來的：
    ma60   ：收盤價在季線（60 日均線）之上才持有（主要規則：網站的「趨勢多空」就是用季線）
    ma240  ：收盤價在年線（240 日均線）之上才持有
    cross  ：月線（20 日）在季線（60 日）之上才持有
    mom120 ：近 120 個交易日報酬為正才持有
  做法：第 t 日收盤後依規則決定，第 t+1 日才生效（position = signal.shift(1)），所以不會用到當天還不知道的資訊。
       不持有時當作現金、報酬為 0。每次進出扣交易成本（台股 0.3%、美股 0.05%，單邊）。
       每檔標的都從「年線可計算」的那天起算，四個規則與「一直持有」比較的是同一段期間。
  事先訂好的三個問題（以各市場等權重組合為準，主要規則 ma60；其餘規則只當參考，因為同時看多個規則容易碰巧有一個好看）：
    Q1 降低大跌損失？  組合的最大回撤小於一直持有，且至少三分之二的標的也是 → 成立（單一歷史路徑，屬描述性）
    Q2 提高風險調整後報酬（夏普值）？ 區塊拔靴法 95% 信賴區間下界 > 0 才算成立
    Q3 提高報酬？      同上，針對年化平均報酬的差距

二、趨勢狀態之後發生什麼（描述性統計）
  「趨勢多」= 收盤價在季線之上且季線比 20 個交易日前高（與網站一致）。比較多／空狀態之後 20 日的報酬與波動。

三、趨勢強度（波動調整）
  強度 = 近 60 日累積對數報酬 ÷（每日波動 × √60）。同樣漲 5%，平常很平靜的股票比平常就很震盪的股票更「不尋常」。
  每日波動用 EWMA（λ=0.94）。這是描述現況的尺度，不是預測。
"""
import json
import os

import numpy as np
import pandas as pd

import core

TREND_JSON = os.path.join(core.DATA_DIR, 'trend.json')
RULES = {'ma60': '站上季線才持有', 'ma240': '站上年線才持有', 'cross': '月線在季線之上才持有', 'mom120': '近 120 日報酬為正才持有'}
PRIMARY = 'ma60'
COST = {'tw': 0.003, 'us': 0.0005}          # 單邊交易成本
FWD = 20
N_BOOT = 2000
STRENGTH_N = 60
BUCKETS = [(-np.inf, -1.5, '強勢下跌'), (-1.5, -0.5, '溫和下跌'), (-0.5, 0.5, '盤整'), (0.5, 1.5, '溫和上升'), (1.5, np.inf, '強勢上升')]


def signals(s):
    """各規則在每個交易日收盤後的訊號（True = 持有）。只用到當日為止的收盤價。"""
    ma20, ma60, ma240 = s.rolling(20).mean(), s.rolling(60).mean(), s.rolling(240).mean()
    return pd.DataFrame({'ma60': s > ma60, 'ma240': s > ma240, 'cross': ma20 > ma60,
                         'mom120': s.pct_change(120, fill_method=None) > 0})[list(RULES)]


def trend_up(s):
    """網站的趨勢多空定義：收盤價在季線之上，且季線比 20 個交易日前高。"""
    ma = s.rolling(60).mean()
    return (s > ma) & (ma > ma.shift(20))


def strength(s, n=STRENGTH_N):
    """波動調整後的趨勢強度（序列）。"""
    r = np.log(s).diff()
    vol = np.sqrt((r ** 2).ewm(alpha=0.06, adjust=False).mean())
    return r.rolling(n).sum() / (vol * np.sqrt(n))


def strength_word(z):
    if z is None or z != z:
        return '資料不足'
    for lo, hi, name in BUCKETS:
        if lo <= z < hi:
            return name
    return '盤整'


def _stats(r):
    r = r.dropna()
    if len(r) < 60:
        return None
    eq = (1 + r).cumprod()
    sd = r.std()
    return {'cagr': float(eq.iloc[-1] ** (core.TRADING_DAYS / len(r)) - 1), 'vol': float(sd * np.sqrt(core.TRADING_DAYS)),
            'sharpe': float(r.mean() / sd * np.sqrt(core.TRADING_DAYS)) if sd > 0 else None,
            'mdd': float((eq / eq.cummax() - 1).min()), 'total': float(eq.iloc[-1] - 1)}


def backtest_one(s, cost):
    """回傳（每日報酬表：bh 與各規則, 各規則的持有比例與進出次數）。"""
    r = s.pct_change(fill_method=None)
    sig = signals(s)
    start = s.rolling(240).mean().first_valid_index()         # 年線可計算後才開始，四個規則期間一致
    if start is None:
        return None, None
    pos = sig.shift(1).astype('boolean').fillna(False).astype(float)   # 前一日收盤後的決定，今天才生效
    out = pd.DataFrame({'bh': r})
    info = {}
    for k in RULES:
        turn = pos[k].diff().abs().fillna(0)
        out[k] = pos[k] * r - turn * cost
        w = pos[k][pos.index > start]
        info[k] = {'in_market': float(w.mean()), 'switches': int(turn[turn.index > start].sum())}
    return out[out.index > start], info


def _block_ci(fn, a, b, block=20, n_boot=N_BOOT, seed=0):
    """成對序列 (a, b) 的統計量差距 fn(a) - fn(b) 的 95% 區塊拔靴信賴區間。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    if n < block * 5:
        return None, None
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([np.arange(s0, s0 + block) for s0 in rng.integers(0, n - block + 1, size=k)])[:n]
        vals.append(fn(a[idx]) - fn(b[idx]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _sharpe(x):
    sd = x.std()
    return x.mean() / sd * np.sqrt(core.TRADING_DAYS) if sd > 0 else 0.0


def _ann_mean(x):
    return x.mean() * core.TRADING_DAYS


def market_backtest(closes, cols, cost):
    per, daily = {}, {}
    for t in cols:
        s = closes[t].dropna()
        d, info = backtest_one(s, cost)
        if d is None or len(d) < 250:
            continue
        daily[t] = d
        per[t] = {'from': d.index[0].strftime('%Y-%m-%d'), 'bh': _stats(d['bh']),
                  'rules': {k: {**_stats(d[k]), **info[k]} for k in RULES}}
    if not daily:
        return None
    # 等權重組合：每天把當天有資料的標的報酬取平均
    port = pd.DataFrame({k: pd.DataFrame({t: d[k] for t, d in daily.items()}).mean(axis=1) for k in ['bh'] + list(RULES)}).dropna()
    res = {'n_tickers': len(daily), 'from': port.index[0].strftime('%Y-%m-%d'), 'to': port.index[-1].strftime('%Y-%m-%d'),
           'cost': cost, 'portfolio': {k: _stats(port[k]) for k in port.columns}, 'per_ticker': per, 'tests': {}}
    for k in RULES:
        better_mdd = sum(1 for v in per.values() if v['rules'][k]['mdd'] > v['bh']['mdd'])      # 回撤是負數，越大（越接近 0）越好
        s_lo, s_hi = _block_ci(_sharpe, port[k].values, port['bh'].values)
        r_lo, r_hi = _block_ci(_ann_mean, port[k].values, port['bh'].values)
        pk, pb = res['portfolio'][k], res['portfolio']['bh']
        res['tests'][k] = {
            'mdd_better_count': better_mdd,
            'q1_less_drawdown': bool(pk['mdd'] > pb['mdd'] and better_mdd >= len(per) * 2 / 3),
            'sharpe_diff': pk['sharpe'] - pb['sharpe'], 'sharpe_ci': [s_lo, s_hi], 'q2_better_sharpe': bool(s_lo is not None and s_lo > 0),
            'ret_diff': float(_ann_mean(port[k].values) - _ann_mean(port['bh'].values)), 'ret_ci': [r_lo, r_hi],
            'q3_better_return': bool(r_lo is not None and r_lo > 0),
            'in_market': float(np.mean([v['rules'][k]['in_market'] for v in per.values()])),
            'switches_per_year': float(np.mean([v['rules'][k]['switches'] for v in per.values()]) / (len(port) / core.TRADING_DAYS))}
    eq = (1 + port).cumprod().resample('W-FRI').last().dropna()
    res['equity'] = {'dates': [d.strftime('%Y-%m-%d') for d in eq.index], **{k: [round(float(x), 4) for x in eq[k]] for k in eq.columns}}
    return res


def state_stats(closes, cols):
    """趨勢多／空狀態之後 20 日的報酬與波動；以及依趨勢強度分組的統計。皆為描述性。"""
    rows = []
    for t in cols:
        s = closes[t].dropna()
        if len(s) < 400:
            continue
        r = np.log(s).diff()
        d = pd.DataFrame({'up': trend_up(s), 'z': strength(s),
                          'fwd': s.shift(-FWD) / s - 1,
                          'fvol': np.sqrt((r ** 2).rolling(FWD).mean().shift(-FWD)) * np.sqrt(core.TRADING_DAYS)})
        d = d[s.rolling(60).mean().shift(20).notna()]
        rows.append(d.dropna())
    if not rows:
        return None
    d = pd.concat(rows).sort_index()

    def grp(x):
        return {'n': int(len(x)), 'ret': float(x['fwd'].mean()), 'up_rate': float((x['fwd'] > 0).mean()), 'vol': float(x['fvol'].mean())}
    up, dn = d[d['up']], d[~d['up']]
    by_date = d.groupby(level=0).apply(lambda x: pd.Series({
        'ret': x.loc[x['up'], 'fwd'].mean() - x.loc[~x['up'], 'fwd'].mean(),
        'vol': x.loc[x['up'], 'fvol'].mean() - x.loc[~x['up'], 'fvol'].mean()})).dropna()
    mean_ci = lambda v: _block_ci(lambda a: a.mean(), v, np.zeros(len(v)), block=2 * FWD)
    out = {'up': grp(up), 'down': grp(dn), 'ret_diff_ci': list(mean_ci(by_date['ret'].values)),
           'vol_diff_ci': list(mean_ci(by_date['vol'].values)), 'buckets': []}
    for lo, hi, name in BUCKETS:
        x = d[(d['z'] >= lo) & (d['z'] < hi)]
        if len(x) >= 50:
            out['buckets'].append({'name': name, **grp(x)})
    return out


def current(m):
    """每檔標的目前的趨勢強度與狀態。"""
    out = {}
    for t in m.tw_cols + m.us_cols:
        s = m.close_all[t].dropna()
        if len(s) < 120:
            continue
        z = strength(s)
        up = trend_up(s)
        zl = float(z.iloc[-1]) if z.iloc[-1] == z.iloc[-1] else None
        out[t] = {'as_of': s.index[-1].strftime('%Y-%m-%d'), 'z60': zl, 'word': strength_word(zl), 'up': bool(up.iloc[-1]),
                  'ret60': float(s.iloc[-1] / s.iloc[-61] - 1) if len(s) > 61 else None,
                  'rules': {k: bool(v) for k, v in signals(s).iloc[-1].items()}}
    return out


def build_all(m):
    res = {'updated_at': pd.Timestamp.now(tz='Asia/Taipei').strftime('%Y-%m-%d %H:%M'), 'rules': RULES, 'primary': PRIMARY,
           'fwd': FWD, 'backtest': {}, 'states': {}, 'current': current(m), 'status': []}
    for market, closes, cols, mname in (('tw', m.tw_close, m.tw_cols, '台股'), ('us', m.us_close, m.us_cols, '美股')):
        bt = market_backtest(closes, cols, COST[market])
        res['backtest'][market] = bt
        res['states'][market] = state_stats(closes, cols)
        if bt:
            t = bt['tests'][PRIMARY]
            res['status'].append(f"{mname}季線規則：回撤較小 {'是' if t['q1_less_drawdown'] else '否'}、夏普較高 {'是' if t['q2_better_sharpe'] else '未證明'}、報酬較高 {'是' if t['q3_better_return'] else '未證明'}")
    return res


def save(res):
    os.makedirs(core.DATA_DIR, exist_ok=True)
    with open(TREND_JSON, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


def load():
    if not os.path.exists(TREND_JSON):
        return None
    with open(TREND_JSON, encoding='utf-8') as f:
        return json.load(f)


if __name__ == '__main__':
    r = build_all(core.load_market())
    save(r)
    print('\n'.join(r['status']))
