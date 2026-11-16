"""大盤長歷史波動模型：用 1998 年起約 25 年的台灣加權指數與 S&P 500，檢驗波動預測。

為什麼：`volmodel.py` 的個股模型只有約 5 年資料、樣本外只有 2024 年之後。大盤有 25 年、涵蓋 2000、2008、2020、2022 四次高波動，
訓練資料多得多、驗證也更可信。結果用於 `risk.py` 的部位大小（步驟 3 的 A/B）。

【事先固定的規則】
  目標：未來 h 日（5／10／20）實現波動 = sqrt(平均(每日對數報酬²))，取對數。
  候選：HAR（線性，近 1／5／22／66 日波動與近 1 年平均）。基準：EWMA(0.94)、延續（過去 h 日波動）、長期平均（過去 1 年）。
  驗證：逐年重訓（walk-forward），第 Y 年只用「Y 年之前、且答案在 Y 年 1 月 1 日前已知」的樣本；樣本外從 2004 年起。
  勝過基準：HAR 的對數均方誤差低於 EWMA 與延續中較好者，且以日期為單位的區塊拔靴 95% 區間（區塊 = 2h 日）下界 > 0。
  不調參數、不加特徵；結果不好就如實記錄。
"""
import json
import os

import numpy as np
import pandas as pd

import core
import regime
import trend

OUT_JSON = os.path.join(core.DATA_DIR, 'volindex.json')
HORIZONS = {5: '1 週', 10: '2 週', 20: '4 週'}
FIRST_YEAR = 2004
FLOOR = 5e-4
COLS = ['lv1', 'lv5', 'lv22', 'lv66', 'lmean']


def _lg(x):
    return np.log(x.clip(lower=FLOOR))


def build(s):
    r = np.log(s).diff()
    rv = lambda n: np.sqrt((r ** 2).rolling(n).mean())
    X = pd.DataFrame({'lv1': _lg(r.abs()), 'lv5': _lg(rv(5)), 'lv22': _lg(rv(22)), 'lv66': _lg(rv(66)), 'lmean': _lg(rv(252))})
    ewma = np.sqrt((r ** 2).ewm(alpha=0.06, adjust=False).mean())
    return r, X, ewma


def evaluate(s, h):
    r, X, ewma = build(s)
    y = _lg(np.sqrt((r ** 2).rolling(h).mean().shift(-h)))            # 未來 h 日實現波動（對數）
    base_ew, base_pe = _lg(ewma), _lg(np.sqrt((r ** 2).rolling(h).mean()))
    df = X.assign(y=y, ew=base_ew, pe=base_pe, mean=X['lmean']).dropna()
    preds = []
    for yr in range(FIRST_YEAR, df.index.year.max() + 1):
        te = df[df.index.year == yr]
        cutoff = pd.Timestamp(f'{yr}-01-01')
        pos = df.index.searchsorted(cutoff)
        tr = df.iloc[:max(pos - h, 0)]                                  # 答案在 1 月 1 日前已知的樣本
        if len(tr) < 1000 or te.empty:
            continue
        A = np.c_[np.ones(len(tr)), tr[COLS].values]
        beta = np.linalg.lstsq(A, tr['y'].values, rcond=None)[0]
        p = np.c_[np.ones(len(te)), te[COLS].values] @ beta
        preds.append(pd.DataFrame({'y': te['y'], 'har': p, 'ewma': te['ew'], 'persist': te['pe'], 'mean': te['mean']}, index=te.index))
    P = pd.concat(preds)
    err = {k: (P[k] - P['y']) ** 2 for k in ('har', 'ewma', 'persist', 'mean')}
    mse = {k: float(v.mean()) for k, v in err.items()}
    best = min(('ewma', 'persist'), key=lambda k: mse[k])
    diff = (err[best] - err['har']).values                              # 正 = HAR 比較準
    lo, hi = trend._block_ci(lambda a: a.mean(), diff, np.zeros(len(diff)), block=2 * h)
    rc = float(P['har'].rank().corr(P['y'].rank()))
    rc_b = float(P[best].rank().corr(P['y'].rank()))
    return {'n': int(len(P)), 'from': P.index[0].strftime('%Y-%m-%d'), 'to': P.index[-1].strftime('%Y-%m-%d'), 'mse': mse, 'best_base': best,
            'improve': float(1 - mse['har'] / mse[best]), 'ci': [lo, hi], 'beats_baseline': bool(lo is not None and lo > 0),
            'rank_corr': rc, 'rank_corr_base': rc_b, 'skill_vs_mean': float(1 - mse['har'] / mse['mean'])}


def build_all(long=None):
    long = long if long is not None else regime.load_long()
    res = {'updated_at': pd.Timestamp.now(tz='Asia/Taipei').strftime('%Y-%m-%d %H:%M'), 'first_year': FIRST_YEAR, 'markets': {}, 'status': []}
    if long is None:
        return res
    for t, (name, mk) in regime.INDEXES.items():
        s = long[t].dropna()
        res['markets'][t] = {'name': name, 'results': {str(h): evaluate(s, h) for h in HORIZONS}}
        w = [HORIZONS[h] for h in HORIZONS if res['markets'][t]['results'][str(h)]['beats_baseline']]
        res['status'].append(f"{name}：HAR 勝過基準的期間 = {'、'.join(w) if w else '無'}")
    return res


def save(res):
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


def load():
    if not os.path.exists(OUT_JSON):
        return None
    with open(OUT_JSON, encoding='utf-8') as f:
        return json.load(f)


if __name__ == '__main__':
    r = build_all()
    save(r)
    print('\n'.join(r['status']))
    for t, m in r['markets'].items():
        for h, v in m['results'].items():
            print(t, h, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in v.items() if k in ('n', 'improve', 'ci', 'beats_baseline', 'best_base', 'rank_corr', 'rank_corr_base', 'skill_vs_mean')}, {k: round(x, 3) for k, x in v['mse'].items()})


def har_forecast(s, h=5, bias_correct=False):
    """逐年重訓的 HAR 預測（每日波動，樣本外，2004 起）。每個日期只用當時已知的資料訓練。"""
    r, X, ewma = build(s)
    y = _lg(np.sqrt((r ** 2).rolling(h).mean().shift(-h)))
    df = X.assign(y=y).dropna(subset=COLS)
    out = []
    for yr in range(FIRST_YEAR, df.index.year.max() + 1):
        te = df[df.index.year == yr]
        pos = df.index.searchsorted(pd.Timestamp(f'{yr}-01-01'))
        tr = df.iloc[:max(pos - h, 0)].dropna()
        if len(tr) < 1000 or te.empty:
            continue
        beta = np.linalg.lstsq(np.c_[np.ones(len(tr)), tr[COLS].values], tr['y'].values, rcond=None)[0]
        pred = np.c_[np.ones(len(te)), te[COLS].values] @ beta
        if bias_correct:              # 對數常態校正：E[exp(x)] = exp(mu + sigma^2/2)，sigma^2 取訓練期殘差變異數（只用當時已知資料）
            pred = pred + 0.5 * float(np.var(tr['y'].values - np.c_[np.ones(len(tr)), tr[COLS].values] @ beta))
        out.append(pd.Series(np.exp(pred), index=te.index))
    return pd.concat(out)
