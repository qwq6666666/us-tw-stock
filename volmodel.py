"""波動預測模型：預測未來 1／2／4 週（5／10／20 個交易日）「每天漲跌的典型幅度」，不預測方向。

為什麼做這個：漲跌方向在本專案的驗證中無法預測，但波動有延續性（大震盪後常接著大震盪），是研究上較穩定的規律。
用途是估風險（接下來可能震多大），不是判斷買賣。

預測目標：未來 h 日的實現波動 = sqrt(平均(每日對數報酬²))，取對數後做迴歸。只有收盤價，所以沒有盤中高低點的資訊。

候選模型（事先固定，不事後增減）：
  HAR  ：線性迴歸，用近 1／5／22／66 日波動與長期水準（波動預測的經典基準）
  LGBM ：梯度提升樹，另加報酬、回撤、下跌波動、大盤波動、美股波動、匯率波動
簡單基準：EWMA（λ=0.94，RiskMetrics）、延續（過去 h 日波動）、長期平均。

事先訂好的規則（看結果前就寫死）：
  1. 驗證一律按時間切分，訓練與驗證之間隔開 h 個交易日；台股用到的美股資料一律嚴格早於當日（Market.us_known）。
  2. 每個市場在 HAR 與 LGBM 中選「三個期間平均誤差較低」的一個當正式模型（LGBM 沒有比較好就用較簡單的 HAR）。
  3. 「勝過基準」的定義：誤差（對數波動的均方誤差）低於 EWMA 與延續兩者中較好的那個，
     且以日期為單位的區塊拔靴法（block bootstrap）95% 信賴區間下界 > 0。相鄰日期的答案高度重疊，所以不能用一般的顯著性檢定。
  4. 只有勝過基準的（市場, 期間）才會顯示在網頁總覽。
  5. 高低價特徵（Parkinson 範圍波動，用每日最高／最低價）的 A/B：同一套驗證期與參數，三個期間的平均誤差（取該市場採用的模型）
     較低才採用，否則維持只用收盤價。規則在看結果前訂好。
"""
import json
import os

import numpy as np
import pandas as pd

import chainlog
import core

VOL_JSON = os.path.join(core.DATA_DIR, 'vol.json')
LOG_CSV = os.path.join(core.DATA_DIR, 'vol_log.csv')
RES_CSV = os.path.join(core.DATA_DIR, 'vol_results.csv')
LOG_FIELDS = ['made_at', 'ticker', 'horizon', 'pred', 'base', 'close', 'model_ver']      # pred/base：每日波動（模型／EWMA）
RES_FIELDS = ['made_at', 'ticker', 'horizon', 'resolved_at', 'realized', 'ret']
KEY = ['made_at', 'ticker', 'horizon']
MODEL_VER = 'vol-v1'

HORIZONS = {5: '1 週', 10: '2 週', 20: '4 週'}
FOLD_STARTS = ['2024-01-01', '2024-07-01', '2025-01-01', '2025-07-01', '2026-01-01']
FLOOR = 5e-4                       # 每日波動下限（避免對 0 取對數）
HAR_COLS = ['lv1', 'lv5', 'lv22', 'lv66', 'lmean']
RANGE_COLS = ['lpk5', 'lpk22']                     # Parkinson 範圍波動（用最高／最低價），選用
EXTRA_COLS = ['lv10', 'lewma', 'down22', 'ret5', 'ret20', 'dd', 'mkt_lv22', 'us_lv5', 'us_abs1', 'fx_lv20']
ALL_COLS = HAR_COLS + EXTRA_COLS
LGB_PARAMS = dict(n_estimators=200, learning_rate=0.03, num_leaves=7, min_child_samples=100, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, random_state=0, verbose=-1)
N_BOOT = 1000


def _lg(x):
    return np.log(x.clip(lower=FLOOR))


def _rv(r, n):
    """近 n 日實現波動（每日）：sqrt(平均(報酬²))。"""
    return np.sqrt((r ** 2).rolling(n).mean())


def own_features(s):
    """單一標的的特徵（只用到當日為止的收盤價）。回傳（特徵表, 每日對數報酬）。"""
    r = np.log(s).diff()
    f = pd.DataFrame(index=s.index)
    f['lv1'] = _lg(r.abs())
    for n in (5, 10, 22, 66):
        f[f'lv{n}'] = _lg(_rv(r, n))
    f['lewma'] = _lg(np.sqrt((r ** 2).ewm(alpha=0.06, adjust=False).mean()))      # EWMA，λ = 0.94
    f['lmean'] = f['lv22'].expanding(min_periods=120).mean()                      # 這檔標的到當日為止的長期波動水準
    f['down22'] = _lg(np.sqrt((r.clip(upper=0) ** 2).rolling(22).mean()))         # 只算下跌日的波動
    f['ret5'] = s.pct_change(5, fill_method=None)
    f['ret20'] = s.pct_change(20, fill_method=None)
    f['dd'] = s / s.rolling(252, min_periods=120).max() - 1
    return f, r


def range_features(high, low):
    """Parkinson 範圍波動：sqrt(平均((ln(最高/最低))² / (4 ln2)))，取對數。只用到當日為止的高低價。"""
    pk = np.log(high / low) ** 2 / (4 * np.log(2))
    return pd.DataFrame({'lpk5': _lg(np.sqrt(pk.rolling(5).mean())), 'lpk22': _lg(np.sqrt(pk.rolling(22).mean()))})


def market_features(m, market, hl=None):
    """回傳 {標的: (收盤價, 特徵表, 每日對數報酬)}。"""
    if market == 'tw':
        idx, closes, cols, mkt = m.tw_close.index, m.tw_close, m.tw_cols, '^TWII'
        us_r = np.log(m.us_known['^SOX']).diff()            # 台股交易日開盤前已知的美股收盤（嚴格早於當日）
    else:
        idx, closes, cols, mkt = m.us_close.index, m.us_close, m.us_cols, '^GSPC'
        us_r = np.log(m.us_close['^SOX']).diff()
    com = pd.DataFrame(index=idx)
    com['us_lv5'] = _lg(_rv(us_r, 5))
    com['us_abs1'] = _lg(us_r.abs())
    # 大盤指數偶有缺值：先去掉缺值再算波動，再對齊回交易日（缺值日沿用前一日已知的數字）
    com['mkt_lv22'] = (_lg(_rv(np.log(closes[mkt].dropna()).diff(), 22)).reindex(idx).ffill()
                       if mkt in closes.columns else np.nan)
    fx = m.usdtwd.reindex(idx.union(m.usdtwd.index)).ffill().shift(1).reindex(idx)      # 匯率只用前一個交易日
    com['fx_lv20'] = _lg(_rv(np.log(fx).diff(), 20))
    out = {}
    for t in cols:
        s = closes[t].dropna()
        if len(s) < 300:
            continue
        f, r = own_features(s)
        f = f.join(com.reindex(s.index))[ALL_COLS]
        if hl is not None and t in hl['high'].columns and t in hl['low'].columns:
            f = f.join(range_features(hl['high'][t].reindex(s.index), hl['low'][t].reindex(s.index)))
        else:
            f[RANGE_COLS] = np.nan
        out[t] = (s, f, r)
    return out


def build_dataset(feats, h):
    rows = []
    for t, (s, f, r) in feats.items():
        d = f.copy()
        d['y'] = _lg(_rv(r, h).shift(-h))                      # 答案：t+1 … t+h 的實現波動（對數）
        d['ret'] = np.log(s.shift(-h) / s)                     # 同期間的實際漲跌（檢查區間涵蓋率用）
        d['persist'] = _lg(_rv(r, h))                          # 基準：延續過去 h 日的波動
        d['ticker'] = t
        rows.append(d)
    return pd.concat(rows)


def _cols(kind, use_range):
    return (HAR_COLS if kind == 'har' else ALL_COLS) + (RANGE_COLS if use_range else [])


def _fit_predict(tr, te, kind, use_range=False):
    cols = _cols(kind, use_range)
    if kind == 'har':
        from sklearn.linear_model import LinearRegression
        return LinearRegression().fit(tr[cols], tr['y']).predict(te[cols])
    import lightgbm as lgb
    return lgb.LGBMRegressor(**LGB_PARAMS).fit(tr[cols], tr['y']).predict(te[cols])


def walk_forward(ds, h, use_range=False):
    """按時間切分：回傳逐筆的樣本外預測表（har、lgb 與各基準），沒有足夠資料時回傳 None。"""
    d = ds.dropna(subset=ALL_COLS + (RANGE_COLS if use_range else []) + ['y', 'persist'])
    starts = [pd.Timestamp(x) for x in FOLD_STARTS]
    ends = starts[1:] + [d.index.max() + pd.Timedelta(days=1)]
    outs = []
    for a, b in zip(starts, ends):
        tr = d[d.index < a - pd.Timedelta(days=int(h * 1.5) + 1)]      # 隔開 h 個交易日，訓練用的答案不會跨進驗證期
        te = d[(d.index >= a) & (d.index < b)]
        if len(tr) < 1000 or len(te) < 50:
            continue
        o = te[['ticker', 'y', 'ret', 'persist']].copy()
        o['ewma'], o['mean'] = te['lewma'], te['lmean']
        o['har'] = _fit_predict(tr, te, 'har', use_range)
        o['lgb'] = _fit_predict(tr, te, 'lgb', use_range)
        outs.append(o)
    return pd.concat(outs) if outs else None


def block_bootstrap_ci(diff_by_date, block, n_boot=N_BOOT, seed=0):
    """以日期為單位的移動區塊拔靴法，回傳平均值的 95% 信賴區間。diff_by_date：每個日期的平均誤差差距。"""
    x = np.asarray(diff_by_date, dtype=float)
    n = len(x)
    if n < block * 3:
        return None, None
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, k))
    means = np.array([np.concatenate([x[s:s + block] for s in row])[:n].mean() for row in starts])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate_oof(o, h, model):
    """樣本外成績：各方法的均方誤差、相對最佳基準的改善、信賴區間、區間涵蓋率。"""
    mse = {k: float(((o[k] - o['y']) ** 2).mean()) for k in ('har', 'lgb', 'ewma', 'persist', 'mean')}
    best_base = min(('ewma', 'persist'), key=lambda k: mse[k])
    loss_diff = ((o[best_base] - o['y']) ** 2 - (o[model] - o['y']) ** 2).groupby(level=0).mean().sort_index()
    lo, hi = block_bootstrap_ci(loss_diff.values, block=max(2 * h, 10))
    sigma = np.exp(o[model]) * np.sqrt(h)                       # 模型預估的 h 日漲跌「一倍範圍」
    res = {'n': int(len(o)), 'n_dates': int(loss_diff.size), 'mse': mse, 'model': model, 'best_base': best_base,
           'improve': float(1 - mse[model] / mse[best_base]),   # 誤差比最佳基準少幾 %
           'ci_low': lo, 'ci_high': hi,
           'rank_corr': float(o[model].corr(o['y'], method='spearman')),
           'rank_corr_base': float(o[best_base].corr(o['y'], method='spearman')),
           'cover1': float((o['ret'].abs() <= sigma).mean()), 'cover2': float((o['ret'].abs() <= 2 * sigma).mean())}
    # 描述性數字（看到第一次結果後才加，不影響上面事先訂好的通過標準）：相對「只猜長期平均、完全不看近況」的改善
    d_mean = ((o['mean'] - o['y']) ** 2 - (o[model] - o['y']) ** 2).groupby(level=0).mean().sort_index()
    mlo, mhi = block_bootstrap_ci(d_mean.values, block=max(2 * h, 10))
    res['skill_vs_mean'] = float(1 - mse[model] / mse['mean'])
    res['skill_ci_low'], res['skill_ci_high'] = mlo, mhi
    res['has_skill'] = bool(mlo is not None and mlo > 0)
    res['beats_baseline'] = bool(lo is not None and lo > 0)
    return res


def predict_latest(ds, feats, h, model, use_range=False):
    """用全部已知答案訓練，預測每檔標的最新一天之後 h 日的波動。"""
    need = ALL_COLS + (RANGE_COLS if use_range else [])
    d = ds.dropna(subset=need + ['y'])
    out = {}
    latest = []
    for t, (s, f, r) in feats.items():
        x = f[need].dropna()
        if x.empty or x.index[-1] != s.index[-1]:
            continue
        latest.append((t, s, r, x.iloc[[-1]]))
    if not latest:
        return out
    X = pd.concat([x for _, _, _, x in latest])
    pred = _fit_predict(d, X, model, use_range)
    for (t, s, r, x), p in zip(latest, pred):
        daily = float(np.exp(p))
        hist = _rv(r, h).dropna().tail(756)                    # 這檔標的近三年、同樣長度的實現波動分布
        pct = float((hist < daily).mean()) if len(hist) > 100 else None
        out[t] = {'as_of': s.index[-1].strftime('%Y-%m-%d'), 'close': float(s.iloc[-1]), 'daily': daily,
                  'ewma': float(np.exp(x['lewma'].iloc[0])), 'range': daily * float(np.sqrt(h)),
                  'annual': daily * float(np.sqrt(core.TRADING_DAYS)), 'pct': pct,
                  'recent5': float(np.exp(x['lv5'].iloc[0])), 'recent22': float(np.exp(x['lv22'].iloc[0]))}
    return out


def level_word(pct):
    if pct is None:
        return '資料不足'
    return '平靜' if pct < 0.25 else '正常' if pct < 0.75 else '偏高' if pct < 0.90 else '劇烈'


def build_all(m, hl=None):
    """hl：universe.load() 的結果（含 high、low），可為 None。有的話對每個市場做高低價特徵的 A/B，依事先固定的規則決定是否採用。"""
    res = {'updated_at': pd.Timestamp.now(tz='Asia/Taipei').strftime('%Y-%m-%d %H:%M'), 'model_ver': MODEL_VER,
           'horizons': {str(h): n for h, n in HORIZONS.items()}, 'validation': {}, 'chosen': {}, 'predictions': {}, 'status': [],
           'range_ab': {}}
    for market, mname in (('tw', '台股'), ('us', '美股')):
        feats = market_features(m, market, hl)
        dsets = {h: build_dataset(feats, h) for h in HORIZONS}

        def run(use_range):
            oofs = {h: walk_forward(dsets[h], h, use_range) for h in HORIZONS}
            ok = [h for h in HORIZONS if oofs[h] is not None]
            avg = lambda k: float(np.mean([((oofs[h][k] - oofs[h]['y']) ** 2).mean() for h in ok])) if ok else np.inf
            model = 'lgb' if avg('lgb') < avg('har') else 'har'       # 規則 2：LGBM 沒有比較好就用較簡單的 HAR
            return oofs, model, avg(model), {'mse_har': avg('har'), 'mse_lgb': avg('lgb')}

        base = run(False)
        use_range, chosen = False, base
        if hl is not None:
            with_r = run(True)
            use_range = with_r[2] < base[2]                         # 規則 5：平均誤差較低才採用高低價特徵
            chosen = with_r if use_range else base
            res['range_ab'][market] = {'used': bool(use_range), 'mse_without': base[2], 'mse_with': with_r[2],
                                       'model_without': base[1], 'model_with': with_r[1]}
        oofs, model, _, ms = chosen
        res['chosen'][market] = {'model': model, **ms, 'use_range': bool(use_range)}
        res['validation'][market] = {str(h): (evaluate_oof(oofs[h], h, model) if oofs[h] is not None else None) for h in HORIZONS}
        for h in HORIZONS:
            for t, p in predict_latest(dsets[h], feats, h, model, use_range).items():
                p['market'] = market
                res['predictions'].setdefault(t, {})[str(h)] = p
        n_ok = sum(1 for v in res['validation'][market].values() if v and v['beats_baseline'])
        res['status'].append(f"{mname}：{len(feats)} 檔，採用 {'LightGBM' if model == 'lgb' else 'HAR 線性模型'}"
                             + ('、含高低價特徵' if use_range else '') + f"，{n_ok}/{len(HORIZONS)} 個期間勝過基準")
    return res


def save(res):
    os.makedirs(core.DATA_DIR, exist_ok=True)
    with open(VOL_JSON, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


def load():
    if not os.path.exists(VOL_JSON):
        return None
    with open(VOL_JSON, encoding='utf-8') as f:
        return json.load(f)


# ───────────── 成績單：只能新增的預測紀錄與到期結果 ─────────────
def append_predictions(res):
    """把最新預測寫進紀錄（同時記下當時的 EWMA 基準與收盤價，之後才能公平比較）。回傳新增筆數。"""
    rows = []
    for t, by_h in sorted(res.get('predictions', {}).items()):
        for h, p in sorted(by_h.items(), key=lambda kv: int(kv[0])):
            rows.append({'made_at': p['as_of'], 'ticker': t, 'horizon': str(h), 'pred': f"{p['daily']:.8f}",
                         'base': f"{p['ewma']:.8f}", 'close': f"{p['close']:.6f}", 'model_ver': res.get('model_ver', MODEL_VER)})
    return chainlog.append(LOG_CSV, LOG_FIELDS, rows, KEY)


def resolve_new(close_all):
    """把剛到期的預測，用當時的收盤價算出實現波動，寫進只能新增的結果紀錄。回傳新增筆數。"""
    done = {tuple(r[k] for k in KEY) for r in chainlog.read(RES_CSV)}
    cache, rows = {}, []
    for r in chainlog.read(LOG_CSV):
        key = tuple(r[k] for k in KEY)
        t, h = r['ticker'], int(r['horizon'])
        if key in done or t not in close_all.columns:
            continue
        s = cache.setdefault(t, close_all[t].dropna())
        d = pd.Timestamp(r['made_at'])
        if d not in s.index:
            continue
        pos = s.index.get_loc(d)
        if pos + h >= len(s):
            continue
        lr = np.log(s.iloc[pos:pos + h + 1]).diff().dropna()
        rows.append({'made_at': r['made_at'], 'ticker': t, 'horizon': r['horizon'],
                     'resolved_at': s.index[pos + h].strftime('%Y-%m-%d'),
                     'realized': f"{float(np.sqrt((lr ** 2).mean())):.8f}", 'ret': f"{float(np.log(s.iloc[pos + h] / s.iloc[pos])):.8f}"})
    return chainlog.append(RES_CSV, RES_FIELDS, rows, KEY)


def verify_logs():
    return chainlog.verify(LOG_CSV, LOG_FIELDS), chainlog.verify(RES_CSV, RES_FIELDS)


def scorecard():
    """回傳（逐筆表, 摘要）。只採用已寫進結果紀錄的答案。"""
    log = pd.DataFrame(chainlog.read(LOG_CSV))
    if log.empty:
        return log, {'total': 0, 'resolved': 0, 'rows': []}
    resd = pd.DataFrame(chainlog.read(RES_CSV))
    df = log[LOG_FIELDS].merge(resd[RES_FIELDS] if not resd.empty else pd.DataFrame(columns=RES_FIELDS), on=KEY, how='left')
    for c in ('pred', 'base', 'close', 'realized', 'ret'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['horizon'] = df['horizon'].astype(int)
    df['resolved'] = df['realized'].notna()
    lr = np.log(df['realized'].clip(lower=FLOOR))
    df['err_model'] = (np.log(df['pred']) - lr) ** 2
    df['err_base'] = (np.log(df['base']) - lr) ** 2
    df['in1'] = df['ret'].abs() <= df['pred'] * np.sqrt(df['horizon'])
    df['in2'] = df['ret'].abs() <= 2 * df['pred'] * np.sqrt(df['horizon'])
    rows = []
    for label, sub in [('全部期間', df)] + [(f'{h} 個交易日', df[df['horizon'] == h]) for h in sorted(df['horizon'].unique())]:
        r = sub[sub['resolved']]
        row = {'label': label, 'n': int(len(r))}
        if len(r):
            row.update(err_model=float(r['err_model'].mean()), err_base=float(r['err_base'].mean()),
                       cover1=float(r['in1'].mean()), cover2=float(r['in2'].mean()))
        rows.append(row)
    return df, {'total': int(len(df)), 'resolved': int(df['resolved'].sum()), 'rows': rows}


if __name__ == '__main__':
    import universe
    r = build_all(core.load_market(), universe.load())
    save(r)
    print('\n'.join(r['status']))
