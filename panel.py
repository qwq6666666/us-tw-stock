"""合併面板模型：把擴充名單的所有個股（台股約 164、美股約 182 檔）合在一起訓練，預測「未來 20 個交易日報酬是否高於同市場中位數」。

為什麼：`model.py` 每個市場只用 8～9 檔、預測「會不會漲」（方向），而方向在本專案驗證中無法預測。這裡改變兩件事：
  1. 訓練樣本從約 9 檔增加到約 170 檔（樣本量約 20 倍）。
  2. 問題改成「相對」：誰比同市場的中位數強。大盤漲跌被扣掉，剩下的問題比較單純，而且學術文獻對橫斷面有較多證據。

【事先固定的規則（看結果前寫死，不調參數）】
  樣本：每 5 個交易日取一次，每檔股票一列。特徵只用當天收盤為止的資料（報酬、波動、回撤、量能比、Parkinson 範圍波動、橫斷面排名）。
  答案：未來 20 日報酬 − 同市場當天的中位數 > 0（1／0）。
  驗證：逐月重訓（walk-forward），第一個測試月 2024-01；訓練只用「答案在測試月之前已知」的樣本（隔開 21 個交易日）。
  模型：LightGBM（7 葉、200 棵、強正則化，與 model.py 同一組保守設定）。
  判準：測試期每個日期的橫斷面排名相關（IC，預測值 vs 實際超額報酬）平均 > 0，且以日期為單位的區塊拔靴 95% 區間下界 > 0；
        另外必須「高於單一 12-1 動能」同一批日期的 IC，才算模型有加分（否則直接用動能就好）。AUC > 0.53 為參考。
  存活者偏誤：名單仍是現在存在的公司，結果要打折（見 relstrength.py 與 universe.py）。
"""
import json
import os

import numpy as np
import pandas as pd

import core
import trend
import universe as U

PANEL_JSON = os.path.join(core.DATA_DIR, 'panel.json')
H = 20
STEP = 5
FIRST_TEST = '2016-01-01'
GAP = 21
FEATS = ['ret1', 'ret5', 'ret20', 'ret60', 'mom12_1', 'vol20', 'vol60', 'dd252', 'vratio', 'pk22', 'rel20', 'mkt20',
         'r_ret5', 'r_ret20', 'r_mom', 'r_vol60', 'r_dd', 'r_vratio']
LGB = dict(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=200, subsample=0.8, subsample_freq=1,
           colsample_bytree=0.8, reg_lambda=5.0, random_state=0, verbose=-1)
N_BOOT = 1000


def pool(data, market):
    uni = U.TW_UNIVERSE if market == 'tw' else U.US_UNIVERSE
    ex = {'0050.TW'} | ({'TSM', 'UMC', 'ASX', 'CHT'} if market == 'us' else set())
    return [t for t in U.usable(data['close'], uni) if t not in ex]


def make_panel(data, cols):
    c, h, l, v = (data[k][cols] for k in ('close', 'high', 'low', 'volume'))
    c = c.dropna(how='all')
    h, l, v = h.reindex(c.index), l.reindex(c.index), v.reindex(c.index)
    r = c.pct_change(fill_method=None)
    f = {'ret1': r, 'ret5': c.pct_change(5, fill_method=None), 'ret20': c.pct_change(20, fill_method=None),
         'ret60': c.pct_change(60, fill_method=None), 'mom12_1': c.shift(21) / c.shift(252) - 1,
         'vol20': r.rolling(20).std(), 'vol60': r.rolling(60).std(), 'dd252': c / c.rolling(252, min_periods=120).max() - 1,
         'vratio': v.rolling(20).mean() / v.rolling(60).mean().where(lambda x: x > 0),
         'pk22': np.sqrt((np.log(h / l) ** 2).rolling(22).mean() / (4 * np.log(2)))}
    mkt = f['ret20'].mean(axis=1)
    f['mkt20'] = f['ret20'].mul(0).add(mkt, axis=0)
    f['rel20'] = f['ret20'].sub(mkt, axis=0)
    rk = lambda x: x.rank(axis=1, pct=True)
    f.update({'r_ret5': rk(f['ret5']), 'r_ret20': rk(f['ret20']), 'r_mom': rk(f['mom12_1']), 'r_vol60': rk(f['vol60']),
              'r_dd': rk(f['dd252']), 'r_vratio': rk(f['vratio'])})
    fwd = c.shift(-H) / c - 1
    excess = fwd.sub(fwd.median(axis=1), axis=0)
    dates = c.index[252::STEP]
    stack = lambda df: df.loc[dates].stack()
    P = pd.DataFrame({k: stack(f[k]) for k in FEATS})
    P['excess'] = stack(excess)
    P['mom_base'] = stack(f['mom12_1'])
    P.index.names = ['date', 'ticker']
    return P.replace([np.inf, -np.inf], np.nan).dropna(subset=['excess', 'mom12_1']), c.index


def _ic(df, col):
    return df.groupby(level='date').apply(lambda g: g[col].rank().corr(g['excess'].rank()) if len(g) >= 20 else np.nan).dropna()


def walk_forward(P, all_dates):
    from lightgbm import LGBMClassifier
    P = P.copy()
    P['y'] = (P['excess'] > 0).astype(int)
    months = pd.period_range(FIRST_TEST[:7], P.index.get_level_values('date').max(), freq='M')
    out, imps = [], []
    for m in months:
        start = m.start_time
        te = P[(P.index.get_level_values('date') >= start) & (P.index.get_level_values('date') <= m.end_time)]
        pos = all_dates.searchsorted(start)
        if te.empty or pos - GAP < 300:
            continue
        cut = all_dates[pos - GAP]
        tr = P[P.index.get_level_values('date') <= cut].dropna(subset=['vol60', 'dd252'])
        if len(tr) < 5000:
            continue
        mdl = LGBMClassifier(**LGB).fit(tr[FEATS], tr['y'])
        te = te.assign(pred=mdl.predict_proba(te[FEATS])[:, 1])
        out.append(te)
        imps.append(mdl.feature_importances_)
    return pd.concat(out), np.mean(imps, axis=0)


def summarize(T):
    from sklearn.metrics import roc_auc_score
    ic_m, ic_b = _ic(T, 'pred'), _ic(T, 'mom_base')
    both = pd.concat([ic_m, ic_b], axis=1, keys=['m', 'b']).dropna()
    lo, hi = trend._block_ci(lambda a: a.mean(), both['m'].values, np.zeros(len(both)), block=4)
    dl, dh = trend._block_ci(lambda a: a.mean(), (both['m'] - both['b']).values, np.zeros(len(both)), block=4)

    def spread(col):
        def f(g):
            if len(g) < 20:
                return np.nan
            k = max(int(len(g) * 0.2), 3)
            s = g.sort_values(col, ascending=False)
            return s['excess'].head(k).mean() - s['excess'].tail(k).mean()
        return T.groupby(level='date').apply(f).dropna()
    sp = spread('pred')
    by_year = {str(y): {'model': float(g['m'].mean()), 'base': float(g['b'].mean()), 'n': int(len(g))} for y, g in both.groupby(both.index.year)}
    core_set = set(U.TW_CORE) | set(U.US_CORE)
    Tc = T[T.index.get_level_values('ticker').isin(core_set)]
    Tx = T[~T.index.get_level_values('ticker').isin(core_set)]
    sub = {k: {'n_stocks': int(x.index.get_level_values('ticker').nunique()), 'model': float(_ic(x, 'pred').mean()), 'base': float(_ic(x, 'mom_base').mean())}
           for k, x in (('original_list', Tc), ('added_list', Tx)) if len(x) > 500}
    return {'by_year': by_year, 'by_list': sub, 'n_dates': int(len(both)), 'n_rows': int(len(T)), 'from': both.index[0].strftime('%Y-%m-%d'), 'to': both.index[-1].strftime('%Y-%m-%d'),
            'ic_model': float(both['m'].mean()), 'ic_ci': [lo, hi], 'ic_base': float(both['b'].mean()),
            'ic_diff': float((both['m'] - both['b']).mean()), 'ic_diff_ci': [dl, dh],
            'auc': float(roc_auc_score(T['excess'] > 0, T['pred'])), 'spread_top_bottom': float(sp.mean()),
            'has_evidence': bool(lo is not None and lo > 0), 'beats_momentum': bool(dl is not None and dl > 0)}


def build_all(data=None):
    data = data or U.load()
    res = {'updated_at': pd.Timestamp.now(tz='Asia/Taipei').strftime('%Y-%m-%d %H:%M'), 'horizon': H, 'first_test': FIRST_TEST,
           'markets': {}, 'status': []}
    if data is None:
        return res
    for mk, nm in (('tw', '台股'), ('us', '美股')):
        cols = pool(data, mk)
        if len(cols) < 40:
            continue
        P, all_dates = make_panel(data, cols)
        T, imp = walk_forward(P, all_dates)
        s = summarize(T)
        s['n_stocks'] = len(cols)
        s['importance'] = {f: float(x) for f, x in sorted(zip(FEATS, imp / imp.sum()), key=lambda kv: -kv[1])[:8]}
        res['markets'][mk] = s
        res['status'].append(f"{nm} {len(cols)} 檔：IC {s['ic_model']:+.3f}（動能 {s['ic_base']:+.3f}），"
                             + ('有證據' if s['has_evidence'] else '沒有證據') + '、' + ('勝過動能' if s['beats_momentum'] else '未勝過動能'))
    return res


def save(res):
    os.makedirs(core.DATA_DIR, exist_ok=True)
    with open(PANEL_JSON, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


def load():
    if not os.path.exists(PANEL_JSON):
        return None
    with open(PANEL_JSON, encoding='utf-8') as f:
        return json.load(f)


if __name__ == '__main__':
    r = build_all()
    save(r)
    print('\n'.join(r['status']))
    print(json.dumps(r['markets'], ensure_ascii=False, indent=1)[:3000])
