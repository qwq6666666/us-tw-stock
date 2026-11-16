"""模型成績單：把每天的判斷寫進「只能新增、不能修改」的紀錄，到期後用真實收盤價對答案。

- 紀錄檔 data/predictions_log.csv 只會在檔尾新增；每一列帶雜湊鏈（hash = SHA-256(前一列 hash + 本列內容)），
  事後改動或刪除任何一列都會讓驗證失敗；再加上 GitHub 的 commit 歷史，可以回頭查證。
- 同一個（判斷日, 標的, 期間）只會記第一次，重跑不會覆蓋。
- 對答案不另存檔：到期日的收盤價直接從 data/prices.csv 算出，結果永遠由紀錄與價格決定。
- 從啟用當天才開始記，不回填歷史（回填就是事後諸葛）。歷史回測見 model.json 的驗證成績。
"""
import csv
import hashlib
import os

import numpy as np
import pandas as pd

import core

LOG_CSV = os.path.join(core.DATA_DIR, 'predictions_log.csv')
FIELDS = ['made_at', 'ticker', 'horizon', 'prob', 'base', 'model_ver', 'prev_hash', 'hash']
RESULTS_CSV = os.path.join(core.DATA_DIR, 'predictions_results.csv')
R_FIELDS = ['made_at', 'ticker', 'horizon', 'resolved_at', 'close_start', 'close_end', 'outcome', 'prev_hash', 'hash']
GENESIS = '0' * 64
MODEL_VER = 'lgbm-shallow-v2'     # 模型設定有實質改動時改這個字串，成績會分開看得出來
RECENT_DAYS = 60                  # 「近期」成績：只看判斷日在最近這幾個日曆天內的紀錄


def _hash(prev, row):
    body = '|'.join(str(row[k]) for k in FIELDS[:6])
    return hashlib.sha256((prev + '|' + body).encode('utf-8')).hexdigest()


def _read_rows():
    if not os.path.exists(LOG_CSV):
        return []
    with open(LOG_CSV, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def append_predictions(model_res):
    """把 model.build_all 的最新預測加進紀錄；已存在的（判斷日, 標的, 期間）不重複記。回傳新增筆數。"""
    rows = _read_rows()
    seen = {(r['made_at'], r['ticker'], r['horizon']) for r in rows}
    prev = rows[-1]['hash'] if rows else GENESIS
    ver = MODEL_VER + ('+inst' if (model_res.get('inst_ab') or {}).get('used') else '')   # 台股模型有採用法人特徵時，版本另記
    new = []
    for t, by_h in sorted(model_res.get('predictions', {}).items()):
        for h, p in sorted(by_h.items(), key=lambda kv: int(kv[0])):
            key = (p['as_of'], t, str(h))
            if key in seen:
                continue
            row = {'made_at': p['as_of'], 'ticker': t, 'horizon': str(h), 'prob': f"{p['prob']:.6f}",
                   'base': f"{p['base']:.6f}", 'model_ver': ver}
            row['prev_hash'] = prev
            row['hash'] = prev = _hash(prev, row)
            new.append(row)
    if new:
        os.makedirs(core.DATA_DIR, exist_ok=True)
        fresh = not os.path.exists(LOG_CSV)
        with open(LOG_CSV, 'a', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator='\n')
            if fresh:
                w.writeheader()
            w.writerows(new)
    return len(new)


def verify_log():
    """檢查雜湊鏈。回傳（是否完整, 筆數, 第一個出問題的列號或 None）。"""
    prev = GENESIS
    rows = _read_rows()
    for i, r in enumerate(rows, 1):
        if r['prev_hash'] != prev or _hash(prev, r) != r['hash']:
            return False, len(rows), i
        prev = r['hash']
    return True, len(rows), None


def _read_results():
    if not os.path.exists(RESULTS_CSV):
        return []
    with open(RESULTS_CSV, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def _rhash(prev, row):
    body = '|'.join(str(row[k]) for k in R_FIELDS[:7])
    return hashlib.sha256((prev + '|' + body).encode('utf-8')).hexdigest()


def verify_results():
    """檢查「結果紀錄」的雜湊鏈。回傳（是否完整, 筆數, 第一個出問題的列號或 None）。"""
    prev = GENESIS
    rows = _read_results()
    for i, r in enumerate(rows, 1):
        if r['prev_hash'] != prev or _rhash(prev, r) != r['hash']:
            return False, len(rows), i
        prev = r['hash']
    return True, len(rows), None


def resolve_new(close_all):
    """把「剛到期」的判斷結果寫進只能新增的結果紀錄（含雜湊鏈）。之後即使資料源修正歷史價格，已記下的結果也不會變。回傳新增筆數。"""
    live = evaluate(close_all, use_stored=False)
    if live.empty:
        return 0
    rows = _read_results()
    seen = {(r['made_at'], r['ticker'], r['horizon']) for r in rows}
    prev = rows[-1]['hash'] if rows else GENESIS
    new = []
    for r in live[live['resolved']].sort_values(['resolved_at', 'ticker', 'horizon']).itertuples():
        key = (r.made_at.strftime('%Y-%m-%d'), r.ticker, str(r.horizon))
        if key in seen:
            continue
        row = {'made_at': key[0], 'ticker': r.ticker, 'horizon': key[2], 'resolved_at': r.resolved_at.strftime('%Y-%m-%d'),
               'close_start': f'{r.close_start:.6f}', 'close_end': f'{r.close_end:.6f}', 'outcome': int(r.outcome)}
        row['prev_hash'] = prev
        row['hash'] = prev = _rhash(prev, row)
        new.append(row)
    if new:
        fresh = not os.path.exists(RESULTS_CSV)
        with open(RESULTS_CSV, 'a', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=R_FIELDS, lineterminator='\n')
            if fresh:
                w.writeheader()
            w.writerows(new)
    return len(new)


def evaluate(close_all, use_stored=True):
    """對答案：回傳每一筆紀錄的結果表。已到期的有 outcome（1=真的漲了）。
    已寫進結果紀錄的，一律採用當時記下的結果；尚未記下的才用目前價格即時計算。"""
    rows = _read_rows()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df['horizon'] = df['horizon'].astype(int)
    df['prob'] = df['prob'].astype(float)
    df['base'] = df['base'].astype(float)
    df['made_at'] = pd.to_datetime(df['made_at'])
    outcome, due, ret, c0, c1 = [], [], [], [], []
    series = {t: close_all[t].dropna() for t in df['ticker'].unique() if t in close_all.columns}
    for t, d, h in zip(df['ticker'], df['made_at'], df['horizon']):
        s = series.get(t)
        o = dd = rr = a0 = a1 = np.nan
        if s is not None and d in s.index:
            pos = s.index.get_loc(d)
            if pos + h < len(s):
                a0, a1 = s.iloc[pos], s.iloc[pos + h]
                rr = a1 / a0 - 1
                o = float(rr > 0)
                dd = s.index[pos + h]
        outcome.append(o)
        due.append(dd)
        ret.append(rr)
        c0.append(a0)
        c1.append(a1)
    df['outcome'], df['resolved_at'], df['ret'] = outcome, pd.to_datetime(due), ret
    df['close_start'], df['close_end'] = c0, c1
    df.attrs['revised'] = 0
    if use_stored:
        stored = {(r['made_at'], r['ticker'], r['horizon']): r for r in _read_results()}
        for i in range(len(df)):
            r = stored.get((df['made_at'].iloc[i].strftime('%Y-%m-%d'), df['ticker'].iloc[i], str(df['horizon'].iloc[i])))
            if r is None:
                continue
            if df['outcome'].iloc[i] == df['outcome'].iloc[i] and df['outcome'].iloc[i] != float(r['outcome']):
                df.attrs['revised'] += 1           # 目前價格算出的答案和當時記下的不同：資料源事後修正過
            a0, a1 = float(r['close_start']), float(r['close_end'])
            df.iloc[i, df.columns.get_loc('close_start')] = a0
            df.iloc[i, df.columns.get_loc('close_end')] = a1
            df.iloc[i, df.columns.get_loc('ret')] = a1 / a0 - 1
            df.iloc[i, df.columns.get_loc('outcome')] = float(r['outcome'])
            df.iloc[i, df.columns.get_loc('resolved_at')] = pd.Timestamp(r['resolved_at'])
    df['resolved'] = df['outcome'].notna()
    df['pred_up'] = df['prob'] >= 0.5
    df['hit'] = np.where(df['resolved'], (df['pred_up'] == (df['outcome'] == 1)).astype(float), np.nan)
    df['brier'] = (df['prob'] - df['outcome']) ** 2
    df['brier_base'] = (df['base'] - df['outcome']) ** 2
    return df


def _stats(d):
    r = d[d['resolved']]
    if r.empty:
        return {'n': 0}
    return {'n': int(len(r)), 'hit': float(r['hit'].mean()), 'always_up': float(r['outcome'].mean()),
            'brier': float(r['brier'].mean()), 'brier_base': float(r['brier_base'].mean())}


def summarize(df):
    """累計與近期成績（依期間與全部），以及機率校準。"""
    if df.empty:
        return {'total': 0, 'resolved': 0, 'pending': 0, 'rows': [], 'calibration': []}
    out = {'total': int(len(df)), 'resolved': int(df['resolved'].sum()), 'pending': int((~df['resolved']).sum()), 'rows': []}
    cutoff = df['made_at'].max() - pd.Timedelta(days=RECENT_DAYS)
    for label, sub in [('全部期間', df)] + [(f'{h} 個交易日', df[df['horizon'] == h]) for h in sorted(df['horizon'].unique())]:
        out['rows'].append({'label': label, 'all': _stats(sub), 'recent': _stats(sub[sub['made_at'] >= cutoff])})
    r = df[df['resolved']].copy()
    cal = []
    if not r.empty:
        r['bin'] = pd.cut(r['prob'], [0, 0.4, 0.5, 0.6, 0.7, 1.0], include_lowest=True)
        for k, g in r.groupby('bin', observed=True):
            cal.append({'range': str(k), 'n': int(len(g)), 'pred': float(g['prob'].mean()), 'actual': float(g['outcome'].mean())})
    out['calibration'] = cal
    return out
