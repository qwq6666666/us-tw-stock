"""公司前景模型（新聞面）：用歷史新聞做「不偷看未來」的模擬，驗證新聞能不能預測個股之後的漲跌。

模擬的規則（任何一條破壞都會讓結果失真）：
1. 判斷日 d 只能用「發布時間早於 d 當天收盤」的新聞。台股收盤 13:30（台北）、美股 16:00（紐約）；收盤後的新聞算下一個交易日。
2. 答案是判斷日收盤起算 N 個交易日後的漲跌（預設 5 日）。
3. 模型按月滾動：只用「早於該月、且隔開答案期間」的樣本訓練，預測該月，之後才公布答案；訓練時絕不看見該月以後的任何資料。
4. 一律與基準比較（永遠猜漲／猜歷史上漲比例）。模型是詞彙加權的簡單文字分類器（TF-IDF + 邏輯迴歸），
   用意是先量出「新聞裡有沒有可學的訊號」；之後換成 BERT／LoRA 微調（Colab）時，用同一套模擬來比較是否更好。
輸出存成 data/outlook.json；不給買賣結論。
"""
import json
import os

import numpy as np
import pandas as pd

import core

HISTORY_CSV = os.path.join(core.DATA_DIR, 'news_history.csv')
OUTLOOK_JSON = os.path.join(core.DATA_DIR, 'outlook.json')
HORIZON = 5
FIRST_TEST = '2024-07-01'          # 之前的資料只拿來訓練
MIN_TRAIN = 800
SENT_MIN_DAYS = 7                  # 現況分數：看最近幾個日曆天內的新聞


def code_to_ticker(code, close_cols):
    for t in (code, code + '.TW'):
        if t in close_cols:
            return t
    return None


def load_all_news():
    """歷史回補 + 每日累積的新聞，去重。欄位：time, ticker(逗號分隔代號), title, summary。"""
    parts = []
    for p in (HISTORY_CSV, core.NEWS_CSV):
        if os.path.exists(p):
            d = pd.read_csv(p, parse_dates=['time'], dtype={'ticker': str}).fillna('')
            parts.append(d[['time', 'ticker', 'title', 'summary']])
    if not parts:
        return pd.DataFrame(columns=['time', 'ticker', 'title', 'summary'])
    df = pd.concat(parts, ignore_index=True)
    return df.drop_duplicates(subset=['title', 'time']).sort_values('time').reset_index(drop=True)


def _close_times(index, ticker):
    """每個交易日的收盤時間（台北時間，無時區）。用來判斷新聞在收盤前還是收盤後。"""
    if ticker.endswith('.TW'):
        return pd.DatetimeIndex(index) + pd.Timedelta(hours=13, minutes=30)
    et = (pd.DatetimeIndex(index) + pd.Timedelta(hours=16)).tz_localize('America/New_York', ambiguous='NaT', nonexistent='shift_forward')
    return et.tz_convert('Asia/Taipei').tz_localize(None)


def build_samples(news, close_all, horizon=HORIZON):
    """每筆樣本 = （標的, 判斷日）：當天收盤前新發布的新聞文字，加上之後 N 日的漲跌答案。"""
    rows = []
    for ticker_code, grp in _explode(news):
        t = code_to_ticker(ticker_code, close_all.columns)
        if t is None:
            continue
        s = close_all[t].dropna()
        ct = _close_times(s.index, t)
        pos = ct.searchsorted(pd.DatetimeIndex(grp['time']), side='left')     # 第一個「收盤時間 ≥ 發布時間」的交易日
        grp = grp.assign(pos=pos)
        grp = grp[grp['pos'] < len(s)]
        for p, g in grp.groupby('pos'):
            if p + horizon >= len(s):
                ret, y = np.nan, np.nan                                         # 答案還沒出來
            else:
                ret = s.iloc[p + horizon] / s.iloc[p] - 1
                y = float(ret > 0)
            rows.append({'ticker': t, 'date': s.index[p], 'n_news': len(g), 'ret': ret, 'y': y,
                         'text': ' '.join((g['title'] + '。' + g['summary']).tolist())[:3000],
                         'first_title': g['title'].iloc[0]})
    return pd.DataFrame(rows).sort_values(['date', 'ticker']).reset_index(drop=True)


def _explode(news):
    for code in sorted({c for x in news['ticker'] for c in x.split(',') if c}):
        mask = news['ticker'].str.split(',').map(lambda cs: code in cs)
        yield code, news[mask]


def _make_model():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    return make_pipeline(TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3), min_df=5, max_features=40000, sublinear_tf=True),
                         LogisticRegression(C=0.3, max_iter=300))


def simulate(samples, horizon=HORIZON):
    """按月滾動模擬。回傳（逐筆結果表, 摘要）。逐筆結果的 prob 都是「只用當時以前資料」訓練出來的預測。"""
    from sklearn.metrics import brier_score_loss, roc_auc_score
    d = samples.dropna(subset=['y']).copy()
    months = pd.period_range(pd.Timestamp(FIRST_TEST), d['date'].max(), freq='M')
    outs, folds = [], []
    for mth in months:
        a, b = mth.start_time, mth.end_time
        tr = d[d['date'] < a - pd.Timedelta(days=int(horizon * 1.5) + 2)]
        te = d[(d['date'] >= a) & (d['date'] <= b)]
        if len(tr) < MIN_TRAIN or len(te) < 20 or tr['y'].nunique() < 2:
            continue
        mdl = _make_model().fit(tr['text'], tr['y'])
        p = mdl.predict_proba(te['text'])[:, 1]
        te = te.assign(prob=p, base=float(tr['y'].mean()), month=str(mth))
        outs.append(te)
        folds.append({'month': str(mth), 'n': int(len(te)),
                      'auc': float(roc_auc_score(te['y'], p)) if te['y'].nunique() > 1 else None})
    if not outs:
        return pd.DataFrame(), None
    r = pd.concat(outs)
    y, p, base = r['y'].values, r['prob'].values, r['base'].values
    summ = {'n': int(len(r)), 'from': r['date'].min().strftime('%Y-%m-%d'), 'to': r['date'].max().strftime('%Y-%m-%d'),
            'auc': float(roc_auc_score(y, p)), 'acc': float(((p > 0.5) == (y == 1)).mean()), 'acc_always_up': float(y.mean()),
            'brier': float(brier_score_loss(y, np.clip(p, .01, .99))), 'brier_base': float(brier_score_loss(y, base)),
            'folds': folds}
    q = r['prob'].rank(pct=True)
    summ['top_ret'] = float(r.loc[q >= 0.8, 'ret'].mean())          # 模型最看好的 20%：之後平均報酬
    summ['bottom_ret'] = float(r.loc[q <= 0.2, 'ret'].mean())       # 模型最不看好的 20%
    summ['all_ret'] = float(r['ret'].mean())
    summ['top_up'] = float(r.loc[q >= 0.8, 'y'].mean())
    summ['bottom_up'] = float(r.loc[q <= 0.2, 'y'].mean())
    # 排列檢定：在每個月內把答案隨機打亂 300 次，看「純靠運氣」能得到 ≥ 實際 AUC 的機率（p 值，越小越不像運氣）
    rng = np.random.default_rng(0)
    perm = []
    for _ in range(300):
        yy = r.groupby('month')['y'].transform(lambda s: rng.permutation(s.values)).values
        perm.append(roc_auc_score(yy, p))
    summ['perm_p'] = float((np.sum(np.array(perm) >= summ['auc']) + 1) / (len(perm) + 1))
    summ['perm_auc_95'] = float(np.percentile(perm, 95))
    # 事先訂好的判準（看結果前就固定，不事後調整）：AUC > 0.53、Brier 優於基準、且排列檢定 p < 0.05
    summ['beats_baseline'] = bool(summ['auc'] > 0.53 and summ['brier'] < summ['brier_base'] and summ['perm_p'] < 0.05)
    return r.reset_index(drop=True), summ


def current_outlook(samples, close_all):
    """用全部已知答案訓練，替每檔標的最近 N 天的新聞打「新聞前景分」（=預測 5 日上漲機率）。"""
    d = samples.dropna(subset=['y'])
    if len(d) < MIN_TRAIN:
        return {}
    mdl = _make_model().fit(d['text'], d['y'])
    last = samples['date'].max()
    rec = samples[samples['date'] >= last - pd.Timedelta(days=SENT_MIN_DAYS)]
    out = {}
    for t, g in rec.groupby('ticker'):
        p = mdl.predict_proba(g['text'])[:, 1]
        out[t] = {'score': float(np.mean(p)), 'n_news': int(g['n_news'].sum()), 'base': float(d['y'].mean()),
                  'latest': g.sort_values('date')['first_title'].iloc[-1]}
    return out


def build_all(close_all):
    news = load_all_news()
    res = {'updated_at': pd.Timestamp.now(tz='Asia/Taipei').strftime('%Y-%m-%d %H:%M'), 'horizon': HORIZON,
           'n_news': int(len(news)), 'news_from': None if news.empty else news['time'].min().strftime('%Y-%m-%d'),
           'summary': None, 'replay': [], 'current': {}}
    if news.empty:
        return res
    samples = build_samples(news, close_all)
    res['n_samples'] = int(len(samples))
    sim, summ = simulate(samples)
    res['summary'] = summ
    if summ:
        rp = sim.sort_values('date', ascending=False).head(60)
        res['replay'] = [{'date': r.date.strftime('%Y-%m-%d'), 'ticker': r.ticker, 'title': r.first_title,
                          'prob': float(r.prob), 'ret': float(r.ret), 'up': bool(r.y)} for r in rp.itertuples()]
    res['current'] = current_outlook(samples, close_all)
    return res


def save(res):
    os.makedirs(core.DATA_DIR, exist_ok=True)
    with open(OUTLOOK_JSON, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


def load():
    if not os.path.exists(OUTLOOK_JSON):
        return None
    with open(OUTLOOK_JSON, encoding='utf-8') as f:
        return json.load(f)


if __name__ == '__main__':
    r = build_all(core.load_market().close_all)
    save(r)
    print(r['n_news'], '則新聞', r.get('n_samples'), '筆樣本')
    print(json.dumps(r['summary'], ensure_ascii=False, indent=1)[:1500] if r['summary'] else '樣本不足，無法模擬')
