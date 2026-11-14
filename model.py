"""漲跌機率模型：預測未來 1／2／4 週（5／10／20 個交易日）收盤價高於今天的機率。

- 只用收盤價衍生的特徵：趨勢、動能、波動、美股隔夜（嚴格早於當日的美股收盤價）、匯率。
- 台股、美股各訓練一個梯度提升樹（LightGBM），同一市場內的標的合併訓練。
- 驗證一律按時間切分（walk-forward）：只用「過去」訓練、預測「之後」，訓練與驗證之間隔開預測天數，避免偷看未來。
- 每次都與簡單基準比較（永遠猜漲／永遠猜歷史上漲比例）。結果存成 data/model.json，網頁只讀檔。
- 台股可加入法人買賣超特徵（institutional.py 回補的逐日資料；盤後才公布，所以判斷日 t 只用到 t-1 以前）。
  是否採用由 A/B 驗證決定：同一套驗證期與參數，平均 AUC 較高才採用（規則事先固定）。
輸出的是機率與主要影響因素，不是買賣建議。
"""
import json
import os

import numpy as np
import pandas as pd

import core

MODEL_JSON = os.path.join(core.DATA_DIR, 'model.json')
HORIZONS = {5: '1 週', 10: '2 週', 20: '4 週'}
FOLD_STARTS = ['2024-01-01', '2024-07-01', '2025-01-01', '2025-07-01', '2026-01-01']   # 每段約半年，最後一段到資料結尾
FEATURES = {
    'ret_5':     '近 5 日報酬',
    'ret_20':    '近 20 日報酬',
    'ret_60':    '近 60 日報酬',
    'ret_120':   '近 120 日報酬',
    'ma20_gap':  '股價相對月線',
    'ma60_gap':  '股價相對季線',
    'ma240_gap': '股價相對年線',
    'rsi':       'RSI（過熱／過冷指標）',
    'vol_20':    '近 20 日波動',
    'vol_60':    '近 60 日波動',
    'dd_252':    '距一年高點的跌幅',
    'us_1d':     '費半（美股）前一日報酬',
    'us_5d':     '費半（美股）近 5 日報酬',
    'spx_5d':    'S&P 500 近 5 日報酬',
    'fx_20d':    '美元兌台幣近 20 日變化',
}
INST_FEATURES = {
    'foreign_5d': '外資近 5 日買賣超（相對平常單日量）',
    'trust_5d':   '投信近 5 日買賣超（相對平常單日量）',
    'mkt_inst_5d': '全市場三大法人近 5 日買賣超（相對平常單日量）',
}
LABELS = {**FEATURES, **INST_FEATURES}
BASE_COLS = list(FEATURES)
# 刻意保守（淺樹、強正則化）：2026-11 試過 7 葉／3 葉／2 葉三種，價量特徵在驗證期都沒有勝過基準，
# 因此選機率最貼近歷史基準的一組，不再針對驗證期繼續調參。
LGB_PARAMS = dict(n_estimators=80, learning_rate=0.03, num_leaves=3, min_child_samples=300, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=20.0, random_state=0, verbose=-1)


def _own_features(s):
    r = s.pct_change(fill_method=None)
    f = pd.DataFrame(index=s.index)
    for n in (5, 20, 60, 120):
        f[f'ret_{n}'] = s.pct_change(n, fill_method=None)
    for n in (20, 60, 240):
        f[f'ma{n}_gap'] = s / s.rolling(n).mean() - 1
    f['rsi'] = core.rsi(s)
    f['vol_20'] = r.rolling(20).std()
    f['vol_60'] = r.rolling(60).std()
    f['dd_252'] = s / s.rolling(252, min_periods=120).max() - 1
    return f


def inst_features(inst, idx, code):
    """法人買賣超特徵。盤後才公布 → shift(1)，判斷日 t 只用到 t-1 以前；以「近 5 日合計 ÷ 近 60 日平均單日絕對量」標準化。"""
    def flow(series):
        net = series.reindex(idx).shift(1)
        den = net.abs().rolling(60, min_periods=20).mean()
        return (net.rolling(5, min_periods=3).sum() / den).replace([np.inf, -np.inf], np.nan)
    piv = lambda col: inst.pivot(index='date', columns='code', values=col)
    out = pd.DataFrame(index=idx)
    f, t_, a = piv('foreign'), piv('trust'), piv('total')
    out['foreign_5d'] = flow(f[code]) if code in f.columns else np.nan
    out['trust_5d'] = flow(t_[code]) if code in t_.columns else np.nan
    out['mkt_inst_5d'] = flow(a['ALL']) if 'ALL' in a.columns else np.nan
    return out


def market_features(m, market, inst=None):
    """回傳（各標的的特徵表, 特徵所需的共同欄位）。美股隔夜一律只用嚴格早於當日的資料。"""
    if market == 'tw':
        idx = m.tw_close.index
        known = m.us_known                                    # 台股交易日開盤前已知的美股收盤價
        com = pd.DataFrame(index=idx)
        com['us_1d'] = known['^SOX'].pct_change(fill_method=None)
        com['us_5d'] = known['^SOX'].pct_change(5, fill_method=None)
        com['spx_5d'] = known['^GSPC'].pct_change(5, fill_method=None)
        cols = m.tw_cols
        closes = m.tw_close
    else:
        idx = m.us_close.index
        com = pd.DataFrame(index=idx)
        sox = m.us_close['^SOX']
        com['us_1d'] = sox.pct_change(fill_method=None)
        com['us_5d'] = sox.pct_change(5, fill_method=None)
        com['spx_5d'] = m.us_close['^GSPC'].pct_change(5, fill_method=None)
        cols = m.us_cols
        closes = m.us_close
    fx = m.usdtwd.reindex(idx.union(m.usdtwd.index)).ffill().shift(1).reindex(idx)   # 匯率也只用前一個交易日
    com['fx_20d'] = fx.pct_change(20, fill_method=None)
    out = {}
    for t in cols:
        s = closes[t].dropna()
        if len(s) < 300:
            continue
        f = _own_features(s).join(com.reindex(s.index))
        cols = list(FEATURES)
        if inst is not None and market == 'tw':
            f = f.join(inst_features(inst, s.index, t.split('.')[0]))
            cols += list(INST_FEATURES)
        out[t] = (s, f[cols])
    return out


def build_dataset(feats, h):
    rows = []
    for t, (s, f) in feats.items():
        d = f.copy()
        fwd = s.shift(-h) / s - 1
        d['y'] = (fwd > 0).astype(float).where(fwd.notna())
        d['ticker'] = t
        rows.append(d)
    return pd.concat(rows)


def _metrics(y, p, base_p):
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
    y = np.asarray(y)
    p = np.clip(np.asarray(p), 0.01, 0.99)
    out = {'n': int(len(y)), 'up_rate': float(y.mean()),
           'acc': float(((p > 0.5) == (y == 1)).mean()),
           'acc_always_up': float(y.mean()),
           'brier': float(brier_score_loss(y, p)), 'brier_base': float(brier_score_loss(y, np.full(len(y), base_p))),
           'logloss': float(log_loss(y, p, labels=[0, 1])),
           'logloss_base': float(log_loss(y, np.full(len(y), base_p), labels=[0, 1]))}
    out['auc'] = float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None
    return out


def walk_forward(ds, h, cols=None):
    """按時間切分驗證：每段只用早於該段（且隔開 h 個交易日）的資料訓練。回傳驗證摘要。"""
    import lightgbm as lgb
    cols = cols or BASE_COLS
    d = ds.dropna(subset=BASE_COLS + ['y'])        # 法人特徵允許缺值（2023 年以前沒有資料）
    starts = [pd.Timestamp(x) for x in FOLD_STARTS]
    ends = starts[1:] + [d.index.max() + pd.Timedelta(days=1)]
    oof_y, oof_p, oof_base, folds = [], [], [], []
    for a, b in zip(starts, ends):
        tr = d[d.index < a - pd.Timedelta(days=int(h * 1.5) + 1)]     # 隔開預測天數，訓練標籤才不會跨進驗證期
        te = d[(d.index >= a) & (d.index < b)]
        if len(tr) < 1000 or len(te) < 50:
            continue
        clf = lgb.LGBMClassifier(**LGB_PARAMS).fit(tr[cols], tr['y'])
        p = clf.predict_proba(te[cols])[:, 1]
        base = float(tr['y'].mean())
        mt = _metrics(te['y'], p, base)
        folds.append({'start': a.strftime('%Y-%m'), 'n': mt['n'], 'auc': mt['auc'], 'acc': mt['acc'],
                      'acc_always_up': mt['acc_always_up']})
        oof_y += list(te['y']); oof_p += list(p); oof_base += [base] * len(te)
    if not oof_y:
        return None
    res = _metrics(oof_y, oof_p, float(np.mean(oof_base)))
    # 機率校準：模型說「約 X% 會漲」的那些日子，實際漲的比例
    q = pd.DataFrame({'y': oof_y, 'p': oof_p})
    q['bin'] = pd.cut(q['p'], [0, 0.4, 0.5, 0.6, 0.7, 1.0], include_lowest=True)
    res['calibration'] = [{'range': str(k), 'n': int(len(g)), 'pred': float(g.p.mean()), 'actual': float(g.y.mean())}
                          for k, g in q.groupby('bin', observed=True) if len(g) >= 30]
    res['folds'] = folds
    res['beats_baseline'] = bool(res['auc'] is not None and res['auc'] > 0.53 and res['brier'] < res['brier_base'])
    return res


def _fmt(k, v):
    if v is None or v != v:
        return '—'
    if k == 'rsi':
        return f'{v:.0f}'
    if k in INST_FEATURES:
        return f'{v:+.1f} 倍'
    return f'{v:+.1%}'


def predict_latest(ds, feats, h, cols=None):
    """用全部已知結果訓練，預測每檔標的「最新一天」的上漲機率與主要影響因素。"""
    import lightgbm as lgb
    cols = cols or BASE_COLS
    d = ds.dropna(subset=BASE_COLS + ['y'])
    clf = lgb.LGBMClassifier(**LGB_PARAMS).fit(d[cols], d['y'])
    base = float(d['y'].mean())
    out = {}
    for t, (s, f) in feats.items():
        last = f[cols].dropna(subset=BASE_COLS)
        if last.empty or last.index[-1] != s.index[-1]:
            continue
        x = last.iloc[[-1]]
        p = float(clf.predict_proba(x[cols])[:, 1][0])
        contrib = clf.predict(x[cols], pred_contrib=True)[0][:-1]      # 每個特徵把機率推高或壓低多少（log-odds）
        order = np.argsort(-np.abs(contrib))[:3]
        out[t] = {'as_of': s.index[-1].strftime('%Y-%m-%d'), 'prob': p, 'base': base,
                  'factors': [{'name': LABELS[cols[i]], 'value': _fmt(cols[i], x.iloc[0][cols[i]]),
                               'effect': '推高' if contrib[i] > 0 else '壓低'} for i in order]}
    return out


def build_all(m, inst=None):
    """inst：institutional.load() 的結果（可為 None）。台股若有足夠法人資料，做 A/B 驗證並依事先固定的規則決定是否採用。"""
    res = {'updated_at': pd.Timestamp.now(tz='Asia/Taipei').strftime('%Y-%m-%d %H:%M'),
           'horizons': {str(h): n for h, n in HORIZONS.items()}, 'validation': {}, 'predictions': {}, 'status': [],
           'inst_ab': None}
    if inst is not None and inst['date'].nunique() < 400:
        inst = None                                    # 資料天數太少不可信，不使用
    for market, mname in (('tw', '台股'), ('us', '美股')):
        feats = market_features(m, market, inst if market == 'tw' else None)
        use_inst = market == 'tw' and inst is not None
        all_cols = BASE_COLS + (list(INST_FEATURES) if use_inst else [])
        dsets = {h: build_dataset(feats, h) for h in HORIZONS}
        chosen = all_cols
        if use_inst:
            vb = {h: walk_forward(dsets[h], h, BASE_COLS) for h in HORIZONS}
            vi = {h: walk_forward(dsets[h], h, all_cols) for h in HORIZONS}
            auc = lambda v: float(np.mean([v[h]['auc'] for h in HORIZONS if v[h]]))
            used = auc(vi) > auc(vb)                   # 事先固定的規則：平均 AUC 較高才採用法人特徵
            chosen = all_cols if used else BASE_COLS
            res['inst_ab'] = {'used': bool(used), 'auc_without': auc(vb), 'auc_with': auc(vi),
                              'days': int(inst['date'].nunique()),
                              'detail': {str(h): {'without': {k: vb[h][k] for k in ('auc', 'brier', 'brier_base')},
                                                  'with': {k: vi[h][k] for k in ('auc', 'brier', 'brier_base')}}
                                         for h in HORIZONS if vb[h] and vi[h]}}
            res['validation'][market] = {str(h): (vi[h] if used else vb[h]) for h in HORIZONS}
        else:
            res['validation'][market] = {str(h): walk_forward(dsets[h], h, chosen) for h in HORIZONS}
        for h in HORIZONS:
            for t, p in predict_latest(dsets[h], feats, h, chosen).items():
                res['predictions'].setdefault(t, {})[str(h)] = p
        res['status'].append(f'{mname}：{len(feats)} 檔、{len(HORIZONS)} 個期間' + ('、含法人買賣超' if chosen is not BASE_COLS and use_inst else ''))
    return res


def save(res):
    os.makedirs(core.DATA_DIR, exist_ok=True)
    with open(MODEL_JSON, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


def load():
    if not os.path.exists(MODEL_JSON):
        return None
    with open(MODEL_JSON, encoding='utf-8') as f:
        return json.load(f)


if __name__ == '__main__':
    import institutional
    r = build_all(core.load_market(), institutional.load())
    save(r)
    print('\n'.join(r['status']))
