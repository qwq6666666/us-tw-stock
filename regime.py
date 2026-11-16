"""市場狀態模型：用隱馬可夫模型（HMM）把大盤分成「多頭／盤整／空頭」三種狀態，並用 1998 年起的長歷史檢驗。

為什麼做這個：趨勢規則實測只有 2021 年之後（以多頭為主），檢驗不到長期空頭。這裡把大盤拉長到 1998 年，
涵蓋 2000 網路泡沫、2008 金融海嘯、2020 疫情、2022 升息，看「判斷狀態」對風險控管有沒有幫助。

【模型】
  3 個狀態的高斯 HMM（自己用 numpy 實作，不增加套件）。每天的觀測 = [日報酬(%), 近 20 日實現波動的對數]，訓練時先標準化。
  狀態命名（每次重訓都依同一規則，不看績效）：平均波動最高者 = 「空頭／高波動」；其餘兩個，平均報酬較高者 = 「多頭」，另一個 = 「盤整」。

【不偷看未來（事先固定的規則）】
  1. 逐年重訓（walk-forward）：第 Y 年的狀態只用第 Y-1 年底以前的資料訓練參數；第一個樣本外年度 = 2004。
  2. 每天的狀態用「濾波機率」（只用到當天為止的觀測，不用未來資料回頭修正），取機率最高者。
  3. 以狀態做的策略：當天收盤後判斷、隔天才生效（position = signal.shift(1)）；狀態 = 空頭 → 空手，其餘持有；扣交易成本（台股 0.3%、美股 0.05%，單邊）。
  4. 對照組：一直持有、季線規則（站上 60 日均線才持有）、年線規則（站上 240 日均線才持有），同一段期間、同樣成本。
  5. 判準（看結果前訂好）：
       Q1 降低大跌損失？ 最大回撤小於一直持有 → 成立（單一歷史路徑，描述性）
       Q2 夏普值較高？   與一直持有的夏普差，區塊拔靴法 95% 信賴區間下界 > 0 才算成立
       Q3 報酬較高？     同上，針對年化平均報酬
       Q4 狀態有分辨力？ 空頭狀態之後 20 日的實現波動，高於多頭狀態（描述性）
  6. 全部條件都是看結果前訂的，參數（3 個狀態、20 日波動、每年重訓）不調。

【限制】狀態判斷有延遲（需要幾天資料才確認轉折）；HMM 假設狀態內報酬接近常態，現實不是；
  1998 年起只有約 3 次明確的大空頭，統計力仍有限。輸出是風險狀態的描述，不是買賣訊號。
"""
import json
import os

import numpy as np
import pandas as pd

import core
import trend

REGIME_JSON = os.path.join(core.DATA_DIR, 'regime.json')
LONG_CSV = os.path.join(core.DATA_DIR, 'index_long.csv')
INDEXES = {'^TWII': ('台灣加權指數', 'tw'), '^GSPC': ('S&P 500', 'us')}
LONG_START = '1998-01-01'
FIRST_TEST_YEAR = 2004
K = 3
VOL_WIN = 20
COST = {'tw': 0.003, 'us': 0.0005}
STATE_NAMES = {'bull': '多頭', 'range': '盤整', 'bear': '空頭／高波動'}
N_BOOT = 1000


# ───────────── 資料 ─────────────
def download_long():
    import yfinance as yf
    raw = yf.download(list(INDEXES), start=LONG_START, auto_adjust=True, progress=False)['Close']
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    return raw.dropna(how='all')


def load_long():
    if not os.path.exists(LONG_CSV):
        return None
    return pd.read_csv(LONG_CSV, index_col=0, parse_dates=True)


def refresh_long():
    """下載長歷史並存檔；下載失敗時沿用舊檔。"""
    try:
        d = download_long()
        if len(d) > 3000:
            d.round(4).to_csv(LONG_CSV)
            return d
    except Exception:
        pass
    return load_long()


def features(s):
    r = np.log(s).diff()
    lv = np.log(np.sqrt((r ** 2).rolling(VOL_WIN).mean()).clip(lower=1e-4))
    return pd.DataFrame({'r': r * 100, 'lv': lv}).dropna()


# ───────────── 高斯 HMM（numpy） ─────────────
class HMM:
    def __init__(self, k=K):
        self.k = k

    def _logb(self, x):
        d = x[:, None, :] - self.mu[None]
        return -0.5 * (np.sum(d ** 2 / self.var[None], axis=2) + np.sum(np.log(2 * np.pi * self.var), axis=1)[None])

    def init(self, x):
        order = np.argsort(x[:, 1])                    # 依波動分三群，給確定的初值
        parts = np.array_split(order, self.k)
        self.mu = np.array([x[p].mean(0) for p in parts])
        self.var = np.array([x[p].var(0) + 1e-3 for p in parts])
        self.A = np.full((self.k, self.k), 0.02 / (self.k - 1))
        np.fill_diagonal(self.A, 0.98)
        self.pi = np.full(self.k, 1 / self.k)

    def _forward(self, logb):
        T = len(logb)
        mx = logb.max(1, keepdims=True)
        b = np.exp(logb - mx)
        alpha, c = np.empty((T, self.k)), np.empty(T)
        a = self.pi * b[0]
        c[0] = a.sum()
        alpha[0] = a / c[0]
        for t in range(1, T):
            a = (alpha[t - 1] @ self.A) * b[t]
            c[t] = a.sum()
            alpha[t] = a / c[t]
        return alpha, c, b, float(np.log(c).sum() + mx.sum())

    def fit(self, x, iters=40, warm=False):
        if not warm:
            self.init(x)
        for _ in range(iters):
            logb = self._logb(x)
            alpha, c, b, _ = self._forward(logb)
            T = len(x)
            beta = np.empty_like(alpha)
            beta[-1] = 1
            for t in range(T - 2, -1, -1):
                beta[t] = (self.A @ (b[t + 1] * beta[t + 1])) / c[t + 1]
            gamma = alpha * beta
            gamma /= gamma.sum(1, keepdims=True)
            xi = (alpha[:-1, :, None] * self.A[None] * (b[1:] * beta[1:])[:, None, :] / c[1:, None, None]).sum(0)
            self.pi = gamma[0]
            self.A = xi / xi.sum(1, keepdims=True)
            w = gamma.sum(0)
            self.mu = (gamma.T @ x) / w[:, None]
            self.var = np.einsum('tk,tkd->kd', gamma, (x[:, None, :] - self.mu[None]) ** 2) / w[:, None] + 1e-3
        return self

    def filter(self, x, prior=None):
        """濾波機率：每天只用到當天為止的觀測。prior = 前一天的濾波機率（接續用）。"""
        logb = self._logb(x)
        mx = logb.max(1, keepdims=True)
        b = np.exp(logb - mx)
        out = np.empty((len(x), self.k))
        p = self.pi if prior is None else prior @ self.A
        for t in range(len(x)):
            a = (p if t == 0 else out[t - 1] @ self.A) * b[t]
            out[t] = a / a.sum()
        return out

    def label(self):
        """狀態命名：波動最高 = 空頭；其餘報酬較高 = 多頭，另一個 = 盤整。回傳 {狀態編號: 'bull'/'range'/'bear'}。"""
        bear = int(np.argmax(self.mu[:, 1]))
        rest = [i for i in range(self.k) if i != bear]
        bull = max(rest, key=lambda i: self.mu[i, 0])
        rng = [i for i in rest if i != bull][0]
        return {bear: 'bear', bull: 'bull', rng: 'range'}


def _standardize(f, ref):
    m, s = ref.mean().values, ref.std().values
    return ((f.values - m) / s), m, s


def walk_forward(s, first_year=FIRST_TEST_YEAR, iters_first=60, iters_warm=20):
    """逐年重訓的樣本外狀態。回傳 DataFrame：欄 = bull/range/bear 機率與 state。"""
    f = features(s)
    years = [y for y in sorted(set(f.index.year)) if y >= first_year]
    out, hmm = [], None
    for y in years:
        train = f[f.index < f'{y}-01-01']
        test = f[f.index.year == y]
        if len(train) < 1000 or test.empty:
            continue
        xt, m, sd = _standardize(train, train)
        warm = hmm is not None
        hmm = (hmm if warm else HMM()).fit(xt, iters=iters_warm if warm else iters_first, warm=warm)
        lab = hmm.label()
        # 接續訓練期最後一天的濾波機率，再逐日往前（只用到當天為止的資料）
        ptrain = hmm.filter(xt)[-1]
        xs = (test.values - m) / sd
        post = hmm.filter(xs, prior=ptrain)
        df = pd.DataFrame(0.0, index=test.index, columns=['bull', 'range', 'bear'])
        for i, name in lab.items():
            df[name] = post[:, i]
        df['year_model'] = y
        out.append(df)
    res = pd.concat(out)
    res['state'] = res[['bull', 'range', 'bear']].idxmax(axis=1)
    return res


def fit_current(s):
    """用全部資料訓練、回傳目前的狀態機率與轉移矩陣（只用於描述現況）。"""
    f = features(s)
    x, m, sd = _standardize(f, f)
    hmm = HMM().fit(x, iters=60)
    lab = hmm.label()
    post = hmm.filter(x)[-1]
    probs = {lab[i]: float(post[i]) for i in range(K)}
    trans = {lab[i]: {lab[j]: float(hmm.A[i, j]) for j in range(K)} for i in range(K)}
    desc = {lab[i]: {'daily_ret_pct': float(hmm.mu[i, 0] * sd[0] + m[0]),
                     'daily_vol_pct': float(np.exp(hmm.mu[i, 1] * sd[1] + m[1]) * 100)} for i in range(K)}
    dur = {k: float(1 / max(1 - trans[k][k], 1e-6)) for k in trans}
    return {'as_of': f.index[-1].strftime('%Y-%m-%d'), 'probs': probs, 'state': max(probs, key=probs.get),
            'transition': trans, 'expected_days': dur, 'desc': desc}


# ───────────── 檢驗 ─────────────
def _stats(r):
    return trend._stats(r)


def evaluate(s, states, cost):
    r = s.pct_change(fill_method=None).reindex(states.index)
    ma60, ma240 = s.rolling(60).mean(), s.rolling(240).mean()
    sigs = {'regime': (states['state'] != 'bear'), 'ma60': (s > ma60).reindex(states.index), 'ma240': (s > ma240).reindex(states.index)}
    daily = pd.DataFrame({'bh': r})
    info = {}
    for k, sg in sigs.items():
        pos = sg.shift(1).astype('boolean').fillna(True).astype(float)       # 隔天才生效
        turn = pos.diff().abs().fillna(0)
        daily[k] = pos * r - turn * cost
        info[k] = {'in_market': float(pos.mean()), 'switches_per_year': float(turn.sum() / (len(pos) / core.TRADING_DAYS))}
    daily = daily.iloc[1:].dropna()
    tests = {}
    for k in sigs:
        pk, pb = _stats(daily[k]), _stats(daily['bh'])
        s_lo, s_hi = trend._block_ci(trend._sharpe, daily[k].values, daily['bh'].values)
        r_lo, r_hi = trend._block_ci(trend._ann_mean, daily[k].values, daily['bh'].values)
        tests[k] = {**pk, **info[k], 'q1_less_drawdown': bool(pk['mdd'] > pb['mdd']),
                    'sharpe_diff': pk['sharpe'] - pb['sharpe'], 'sharpe_ci': [s_lo, s_hi], 'q2_better_sharpe': bool(s_lo is not None and s_lo > 0),
                    'ret_diff': float(trend._ann_mean(daily[k].values) - trend._ann_mean(daily['bh'].values)), 'ret_ci': [r_lo, r_hi],
                    'q3_better_return': bool(r_lo is not None and r_lo > 0)}
    # 各狀態之後的表現（描述性）：狀態 t 之後 20 日的報酬與實現波動
    lr = np.log(s).diff()
    fwd = pd.DataFrame({'fwd': s.shift(-20) / s - 1, 'fvol': np.sqrt((lr ** 2).rolling(20).mean().shift(-20)) * np.sqrt(core.TRADING_DAYS),
                        'state': states['state'].reindex(s.index)}).dropna()
    by_state = {}
    for st in ('bull', 'range', 'bear'):
        x = fwd[fwd['state'] == st]
        by_state[st] = {'n': int(len(x)), 'share': float(len(x) / len(fwd)) if len(fwd) else 0,
                        'fwd20_ret': float(x['fwd'].mean()) if len(x) else None, 'fwd20_vol': float(x['fvol'].mean()) if len(x) else None,
                        'up_rate': float((x['fwd'] > 0).mean()) if len(x) else None}
    bear, bull = fwd[fwd['state'] == 'bear'], fwd[fwd['state'] == 'bull']
    q4 = bool(len(bear) > 30 and len(bull) > 30 and bear['fvol'].mean() > bull['fvol'].mean())
    # 持續性：狀態平均連續幾天、一年換幾次
    chg = (states['state'] != states['state'].shift()).sum()
    eq = (1 + daily).cumprod().resample('ME').last().dropna()
    return {'from': daily.index[0].strftime('%Y-%m-%d'), 'to': daily.index[-1].strftime('%Y-%m-%d'), 'n_days': int(len(daily)),
            'bh': _stats(daily['bh']), 'strategies': tests, 'by_state': by_state, 'q4_vol_separates': q4,
            'state_changes_per_year': float(chg / (len(states) / core.TRADING_DAYS)),
            'equity': {'dates': [d.strftime('%Y-%m-%d') for d in eq.index], **{k: [round(float(x), 4) for x in eq[k]] for k in eq.columns}},
            'periods': _bear_periods(states)}


def _bear_periods(states, min_days=10):
    """連續空頭狀態的區間（至少 min_days 個交易日），供畫圖標示。"""
    isb = (states['state'] == 'bear').astype(int)
    grp = (isb != isb.shift()).cumsum()
    out = []
    for _, g in isb.groupby(grp):
        if g.iloc[0] == 1 and len(g) >= min_days:
            out.append([g.index[0].strftime('%Y-%m-%d'), g.index[-1].strftime('%Y-%m-%d')])
    return out


def build_all(long=None):
    long = long if long is not None else load_long()
    res = {'updated_at': pd.Timestamp.now(tz='Asia/Taipei').strftime('%Y-%m-%d %H:%M'), 'first_test_year': FIRST_TEST_YEAR,
           'state_names': STATE_NAMES, 'markets': {}, 'status': []}
    if long is None:
        return res
    for t, (name, mk) in INDEXES.items():
        if t not in long.columns:
            continue
        s = long[t].dropna()
        states = walk_forward(s)
        ev = evaluate(s, states, COST[mk])
        cur = fit_current(s)
        # 最近一年的狀態序列（用樣本外濾波結果）
        recent = states.tail(260)
        res['markets'][t] = {'name': name, 'market': mk, 'history_from': s.index[0].strftime('%Y-%m-%d'), 'current': cur, 'eval': ev,
                             'recent': {'dates': [d.strftime('%Y-%m-%d') for d in recent.index], 'state': recent['state'].tolist(),
                                        'bear_prob': [round(float(x), 3) for x in recent['bear']]}}
        rg, bh = ev['strategies']['regime'], ev['bh']
        res['status'].append(f"{name}（{ev['from'][:4]} 起樣本外）：目前 {STATE_NAMES[cur['state']]}；狀態策略最大回撤 {rg['mdd']:.0%} vs 一直持有 {bh['mdd']:.0%}，"
                             f"夏普{'較高' if rg['q2_better_sharpe'] else '未證明較高'}")
    return res


def save(res):
    os.makedirs(core.DATA_DIR, exist_ok=True)
    with open(REGIME_JSON, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=1)


def load():
    if not os.path.exists(REGIME_JSON):
        return None
    with open(REGIME_JSON, encoding='utf-8') as f:
        return json.load(f)


if __name__ == '__main__':
    d = refresh_long()
    r = build_all(d)
    save(r)
    print('\n'.join(r['status']))
