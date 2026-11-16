"""市場狀態與部位建議的防竄改成績單（只能新增、SHA-256 雜湊鏈，用 chainlog.py）。

每天記下兩種判斷，20 個交易日後對答案：
  regime ：大盤的狀態（多頭／盤整／空頭）與空頭機率。答案 = 之後 20 日的實現年化波動與報酬。
  risk   ：每檔的預測年化波動與建議的波動目標權重。答案 = 之後 20 日的實現年化波動。
成績單的判讀（事先固定）：
  狀態：空頭／高波動狀態之後的實現波動應高於多頭（與歷史實測一致）；樣本太少（到期 < 30 筆）時只顯示「樣本太少」。
  部位：實現波動與預測的比值，與歷史上的涵蓋率比較；不評價報酬。
資料來源只用當時記下的收盤價與預測，之後資料源修正歷史也不改變成績。不可手動編輯。
"""
import numpy as np
import pandas as pd

import chainlog
import core

LOG_CSV = core.DATA_DIR + '/state_log.csv'
RES_CSV = core.DATA_DIR + '/state_results.csv'
LOG_FIELDS = ['made_at', 'kind', 'ticker', 'label', 'value', 'close', 'model_ver']      # value：regime = 空頭機率；risk = 預測年化波動
RES_FIELDS = ['made_at', 'kind', 'ticker', 'resolved_at', 'realized_vol', 'ret']
KEY = ['made_at', 'kind', 'ticker']
H = 20
MODEL_VER = 'state-v1'


def append_daily(regime_res, risk_res, close_all):
    rows = []
    for t, m in (regime_res or {}).get('markets', {}).items():
        cur = m['current']
        if t in close_all.columns and pd.Timestamp(cur['as_of']) in close_all[t].dropna().index:
            rows.append({'made_at': cur['as_of'], 'kind': 'regime', 'ticker': t, 'label': cur['state'], 'value': f"{cur['probs']['bear']:.6f}",
                         'close': f"{close_all[t].dropna().loc[cur['as_of']]:.6f}", 'model_ver': MODEL_VER})
    for t, v in (risk_res or {}).get('current', {}).items():
        if t in close_all.columns and pd.Timestamp(v['as_of']) in close_all[t].dropna().index:
            rows.append({'made_at': v['as_of'], 'kind': 'risk', 'ticker': t, 'label': f"w={v['weight_vt']:.3f}", 'value': f"{v['annual20']:.6f}",
                         'close': f"{close_all[t].dropna().loc[v['as_of']]:.6f}", 'model_ver': MODEL_VER})
    return chainlog.append(LOG_CSV, LOG_FIELDS, rows, KEY)


def resolve_new(close_all):
    done = {tuple(r[k] for k in KEY) for r in chainlog.read(RES_CSV)}
    cache, rows = {}, []
    for r in chainlog.read(LOG_CSV):
        key = tuple(r[k] for k in KEY)
        t = r['ticker']
        if key in done or t not in close_all.columns:
            continue
        s = cache.setdefault(t, close_all[t].dropna())
        d = pd.Timestamp(r['made_at'])
        if d not in s.index:
            continue
        pos = s.index.get_loc(d)
        if pos + H >= len(s):
            continue
        lr = np.log(s.iloc[pos:pos + H + 1]).diff().dropna()
        rows.append({'made_at': r['made_at'], 'kind': r['kind'], 'ticker': t, 'resolved_at': s.index[pos + H].strftime('%Y-%m-%d'),
                     'realized_vol': f"{float(np.sqrt((lr ** 2).mean() * core.TRADING_DAYS)):.8f}",
                     'ret': f"{float(np.log(s.iloc[pos + H] / s.iloc[pos])):.8f}"})
    return chainlog.append(RES_CSV, RES_FIELDS, rows, KEY)


def verify_logs():
    return chainlog.verify(LOG_CSV, LOG_FIELDS), chainlog.verify(RES_CSV, RES_FIELDS)


def scorecard():
    log, res = pd.DataFrame(chainlog.read(LOG_CSV)), pd.DataFrame(chainlog.read(RES_CSV))
    if log.empty:
        return {'total': 0, 'resolved': 0, 'regime': None, 'risk': None}
    df = log.merge(res[RES_FIELDS] if not res.empty else pd.DataFrame(columns=RES_FIELDS), on=KEY, how='left')
    for c in ('value', 'realized_vol', 'ret'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    out = {'total': int(len(df)), 'resolved': int(df['realized_vol'].notna().sum())}
    rg = df[(df['kind'] == 'regime') & df['realized_vol'].notna()]
    out['regime'] = None if rg.empty else {'n': int(len(rg)), 'by_state': {k: {'n': int(len(g)), 'vol': float(g['realized_vol'].mean())} for k, g in rg.groupby('label')}}
    rk = df[(df['kind'] == 'risk') & df['realized_vol'].notna()]
    out['risk'] = None if rk.empty else {'n': int(len(rk)), 'ratio_median': float((rk['realized_vol'] / rk['value']).median()),
                                         'within_1x': float(((rk['realized_vol'] / rk['value']).between(0.5, 1.5)).mean())}
    return out
