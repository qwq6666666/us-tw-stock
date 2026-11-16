"""風險說明與部位大小：把波動預測用在「該放多少錢」，而不是「會漲還是會跌」。

波動比方向好預測（見 volmodel.py），所以這裡的核心想法是：預期震盪越大，部位越小，讓每個部位帶來的風險大致相同（波動目標）。
再和趨勢狀態合在一起，做成「趨勢 × 波動」的四格風險框架。**這是風險的描述，不是買賣建議。**

【歷史實測（規則看結果前固定，不調參數）】
  資料：1998 年起的大盤（台灣加權指數、S&P 500），樣本從 2004 年起（與 regime.py 同一段，涵蓋 2008、2020、2022）。
  波動預測：EWMA（λ=0.94，RiskMetrics）。只用 EWMA 是因為正式模型（volmodel.py）只有 2024 年之後的樣本外預測；
           EWMA 是波動預測的標準基準，因此這裡的結果是「保守估計」（正式模型對美股 1~2 週明顯更準）。
  部位：權重 = min(1, 目標波動 15% ÷ 預測年化波動)，不加槓桿；每週（每 5 個交易日）調整一次；訊號隔天才生效；
       權重變動的部分扣交易成本（台股 0.3%、美股 0.05%，單邊）。
  比較的策略：
    bh        一直持有（100%）
    vt        只做波動目標
    trend     站上季線才持有（trend.py 的主要規則）
    trend_vt  站上季線才持有，持有時再依波動調整部位（趨勢 × 波動）
    bh_scaled 一直持有、但部位固定縮成與 vt 相同的平均持股比例（公平對照：回撤變小有多少只是因為持股比較少？）
  事先訂好的判準：
    Q1 vt 的最大回撤小於「同樣平均持股比例的一直持有」（bh_scaled）→ 波動目標比單純減碼更有效
    Q2 vt 的夏普值與一直持有的差，區塊拔靴法 95% 信賴區間下界 > 0 → 風險調整後報酬較高
    Q3 vt 的最大回撤小於一直持有 → 描述性
    trend_vt 同樣列出，但主要問題只看 vt（避免同時看太多組合而碰巧有一組好看）。

【四格風險框架（現況）】
  趨勢多／空：trend.trend_up（收盤價在季線之上且季線向上）。波動高／低：目前的波動預測在自己歷史的百分位，≥ 67% 算高波動。
  只描述「現在處於哪一格、該格的風險特徵」，不給買賣指令；趨勢規則在本專案的實測中沒有提高夏普值，所以框架的部位大小主要由波動決定。
"""
import json
import os

import numpy as np
import pandas as pd

import core
import regime
import trend
import volmodel

RISK_JSON = os.path.join(core.DATA_DIR, 'risk.json')
TARGET_VOL = 0.15
LOSS_BUDGET = 0.03            # 網頁預設值：你能接受一個月（2 倍標準差的壞情況）損失組合的 3%
REBAL = 5
FIRST_YEAR = 2004
HIGH_PCT = 0.67
STRATS = {'bh': '一直持有', 'vt': '波動目標', 'trend': '站上季線才持有', 'trend_vt': '趨勢 × 波動', 'bh_scaled': '一直持有（同平均持股）'}
QUADRANTS = {
    (True, False): ('趨勢多、波動低', '價格在季線之上、預期震盪不大。部位大小主要由你的風險預算決定。'),
    (True, True): ('趨勢多、波動高', '價格在季線之上，但預期震盪偏大：同樣的風險預算下，波動目標會建議比平常小的部位。'),
    (False, False): ('趨勢空、波動低', '價格在季線之下、預期震盪不大。趨勢規則在本專案實測沒有提高夏普值，部位大小仍以風險預算為主。'),
    (False, True): ('趨勢空、波動高', '價格在季線之下且預期震盪偏大：預期的單月波動幅度最大的一格，部位大小宜用較壞的情況估算。'),
}


def ewma_vol(r, lam=0.94):
    """每日 EWMA 波動（只用到當天為止的資料）。"""
    return np.sqrt((r ** 2).ewm(alpha=1 - lam, adjust=False).mean())


def weights(s, target=TARGET_VOL):
    """依 EWMA 波動的目標權重（當天收盤後決定）。每 REBAL 個交易日更新一次。"""
    r = np.log(s).diff()
    ann = ewma_vol(r) * np.sqrt(core.TRADING_DAYS)
    w = (target / ann).clip(upper=1.0)
    w = w.where(np.arange(len(w)) % REBAL == 0).ffill()
    return w


def backtest(s, cost):
    r = s.pct_change(fill_method=None)
    w = weights(s)
    up = (s > s.rolling(60).mean())
    out = pd.DataFrame({'bh': r})
    pos = {'vt': w, 'trend': up.astype(float), 'trend_vt': w * up}
    for k, p in pos.items():
        p = p.shift(1).fillna(0)                              # 隔天才生效
        out[k] = p * r - p.diff().abs().fillna(0) * cost
    out = out[out.index.year >= FIRST_YEAR].dropna()
    avg = float(pos['vt'].shift(1).reindex(out.index).mean())
    out['bh_scaled'] = out['bh'] * avg
    return out, avg, {k: float(v.shift(1).reindex(out.index).mean()) for k, v in pos.items()}


def evaluate(s, cost):
    d, avg, expo = backtest(s, cost)
    st = {k: trend._stats(d[k]) for k in STRATS}
    s_lo, s_hi = trend._block_ci(trend._sharpe, d['vt'].values, d['bh'].values)
    r_lo, r_hi = trend._block_ci(trend._ann_mean, d['vt'].values, d['bh'].values)
    tv_lo, tv_hi = trend._block_ci(trend._sharpe, d['trend_vt'].values, d['bh'].values)
    # 最壞單月：各策略的月報酬最小值
    m = (1 + d).resample('ME').prod() - 1
    worst = {k: float(m[k].min()) for k in STRATS}
    # 實現波動是否接近目標（只看 vt）
    yr = d['vt'].groupby(d.index.year).std() * np.sqrt(core.TRADING_DAYS)
    eq = (1 + d[['bh', 'vt', 'trend_vt']]).cumprod().resample('ME').last().dropna()
    return {'from': d.index[0].strftime('%Y-%m-%d'), 'to': d.index[-1].strftime('%Y-%m-%d'), 'stats': st, 'exposure': expo,
            'worst_month': worst, 'avg_weight_vt': avg,
            'q1_beats_scaled': bool(st['vt']['mdd'] > st['bh_scaled']['mdd']), 'q3_less_drawdown': bool(st['vt']['mdd'] > st['bh']['mdd']),
            'sharpe_diff': st['vt']['sharpe'] - st['bh']['sharpe'], 'sharpe_ci': [s_lo, s_hi],
            'q2_better_sharpe': bool(s_lo is not None and s_lo > 0),
            'ret_diff': float(trend._ann_mean(d['vt'].values) - trend._ann_mean(d['bh'].values)), 'ret_ci': [r_lo, r_hi],
            'trend_vt_sharpe_diff': st['trend_vt']['sharpe'] - st['bh']['sharpe'], 'trend_vt_sharpe_ci': [tv_lo, tv_hi],
            'realized_vol_by_year': {int(y): float(v) for y, v in yr.items()},
            'equity': {'dates': [x.strftime('%Y-%m-%d') for x in eq.index], **{k: [round(float(x), 4) for x in eq[k]] for k in eq.columns}}}


def quadrant(up, pct):
    key = (bool(up), bool(pct is not None and pct >= HIGH_PCT))
    return {'key': f"{'up' if key[0] else 'down'}_{'high' if key[1] else 'low'}", 'name': QUADRANTS[key][0], 'text': QUADRANTS[key][1]}


def current(vol, tr, names=None):
    """每檔標的目前的風險狀態。vol = volmodel 的輸出、tr = trend 的輸出。"""
    out = {}
    cur_tr = (tr or {}).get('current', {})
    for t, hs in (vol or {}).get('predictions', {}).items():
        p20, p5 = hs.get('20'), hs.get('5')
        if not p20:
            continue
        c = cur_tr.get(t, {})
        up = c.get('up')
        q = quadrant(up, p20.get('pct')) if up is not None else None
        out[t] = {'as_of': p20['as_of'], 'daily20': p20['daily'], 'annual20': p20['annual'], 'pct': p20.get('pct'),
                  'daily5': p5['daily'] if p5 else None, 'trend_up': up, 'quadrant': q,
                  'weight_vt': float(min(1.0, TARGET_VOL / p20['annual'])) if p20['annual'] > 0 else 1.0}
    return out


def build_all(long=None, vol=None, tr=None):
    long = long if long is not None else regime.load_long()
    vol = vol if vol is not None else volmodel.load()
    tr = tr if tr is not None else trend.load()
    res = {'updated_at': pd.Timestamp.now(tz='Asia/Taipei').strftime('%Y-%m-%d %H:%M'), 'target_vol': TARGET_VOL, 'loss_budget': LOSS_BUDGET,
           'rebalance_days': REBAL, 'high_pct': HIGH_PCT, 'strategies': STRATS, 'backtest': {}, 'current': current(vol, tr), 'status': []}
    if long is not None:
        for t, (name, mk) in regime.INDEXES.items():
            if t in long.columns:
                ev = evaluate(long[t].dropna(), regime.COST[mk])
                ev['name'] = name
                res['backtest'][t] = ev
                v, b = ev['stats']['vt'], ev['stats']['bh']
                res['status'].append(f"{name}：波動目標最大回撤 {v['mdd']:.0%} vs 一直持有 {b['mdd']:.0%}"
                                     f"（同平均持股 {ev['stats']['bh_scaled']['mdd']:.0%}）、夏普{'較高' if ev['q2_better_sharpe'] else '未證明較高'}")
    try:
        res['ab_har'] = ab_har(long)          # A/B：HAR 預測是否比 EWMA 更適合決定部位（規則見檔尾）
        res['status'].append('部位 A/B：HAR ' + ('採用' if res['ab_har']['adopt_all'] else '未勝過 EWMA，維持 EWMA'))
    except Exception as e:
        res['status'].append(f'部位 A/B：失敗（{type(e).__name__}）')
    return res


def save(res):
    os.makedirs(core.DATA_DIR, exist_ok=True)
    with open(RISK_JSON, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


def load():
    if not os.path.exists(RISK_JSON):
        return None
    with open(RISK_JSON, encoding='utf-8') as f:
        return json.load(f)


if __name__ == '__main__':
    r = build_all()
    save(r)
    print('\n'.join(r['status']))


# ───────────── A/B：用 HAR 預測取代 EWMA 決定部位（規則事先固定） ─────────────
# 採用條件（兩個指數都要成立）：(1) 實現波動與目標的平均絕對差（逐年）比 EWMA 小；(2) 最大回撤不比 EWMA 差超過 2 個百分點。
# 第二個 A/B（2026-11-17 事先登記，規則同上）：HAR 加對數常態偏誤修正 exp(mu + σ²/2)，σ² 取訓練期殘差變異數；只在這一次比較，不再調整修正方式。
# 預測期間用 5 日（與每週調整一致）。HAR 由 volindex.py 逐年重訓產生，只用當時已知的資料。
def ab_har(long=None):
    import volindex
    long = long if long is not None else regime.load_long()
    out = {}
    for t, (name, mk) in regime.INDEXES.items():
        s = long[t].dropna()
        r = s.pct_change(fill_method=None)
        har = volindex.har_forecast(s, 5)
        har_bc = volindex.har_forecast(s, 5, bias_correct=True)
        res = {}
        for label, vol in (('ewma', ewma_vol(np.log(s).diff())), ('har', har), ('har_bc', har_bc)):
            ann = vol * np.sqrt(core.TRADING_DAYS)
            w = (TARGET_VOL / ann).clip(upper=1.0)
            w = w.where(np.arange(len(w)) % REBAL == 0).ffill()
            p = w.shift(1)
            d = (p * r - p.diff().abs().fillna(0) * regime.COST[mk]).dropna()
            d = d[d.index.year >= FIRST_YEAR]
            d = d[d.index.isin(har.index)]
            yr = d.groupby(d.index.year).std() * np.sqrt(core.TRADING_DAYS)
            st_ = trend._stats(d)
            m = (1 + d).resample('ME').prod() - 1
            res[label] = {'mdd': st_['mdd'], 'sharpe': st_['sharpe'], 'cagr': st_['cagr'], 'worst_month': float(m.min()),
                          'track_gap': float((yr - TARGET_VOL).abs().mean()), 'avg_weight': float(p.reindex(d.index).mean())}
        better = lambda k: bool(res[k]['track_gap'] < res['ewma']['track_gap'] and res[k]['mdd'] > res['ewma']['mdd'] - 0.02)
        out[t] = {'name': name, **res, 'adopt': better('har'), 'adopt_bc': better('har_bc')}
    out['adopt_all'] = bool(all(v['adopt'] for k, v in out.items() if k not in ('adopt_all', 'adopt_all_bc')))
    out['adopt_all_bc'] = bool(all(v['adopt_bc'] for k, v in out.items() if k not in ('adopt_all', 'adopt_all_bc')))
    return out
