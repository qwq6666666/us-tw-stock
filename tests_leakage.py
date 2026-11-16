"""防偷看未來測試：把資料截到某一天，與完整資料比較，截止日以前的輸出必須完全相同。
執行：python tests_leakage.py（全部通過才算過；任何一項失敗代表某個特徵用到了未來的資料）。"""
import numpy as np
import pandas as pd

import regime
import relstrength
import risk
import trend


def _same(a, b, name):
    a, b = a.align(b, join='inner')
    if isinstance(a, pd.DataFrame):
        ok = np.allclose(a.select_dtypes('number').fillna(-9).values, b.select_dtypes('number').fillna(-9).values, atol=1e-9) \
            and (a.select_dtypes(exclude='number').values == b.select_dtypes(exclude='number').values).all()
    else:
        ok = np.allclose(a.fillna(-9).values, b.fillna(-9).values, atol=1e-9)
    print(('PASS ' if ok else 'FAIL ') + name)
    return ok


def main():
    long = regime.load_long()
    s = long['^GSPC'].dropna()
    cut = s.index[-600]
    s_cut = s[s.index <= cut]
    ok = True
    # 1. 市場狀態（逐年重訓＋濾波）
    ok &= _same(regime.walk_forward(s_cut), regime.walk_forward(s).loc[:cut], 'regime.walk_forward 截斷 = 完整（截止日以前）')
    # 2. 波動目標權重
    ok &= _same(risk.weights(s_cut), risk.weights(s).loc[:cut], 'risk.weights')
    # 3. 趨勢訊號與強度
    ok &= _same(trend.signals(s_cut).astype(float), trend.signals(s).loc[:cut].astype(float), 'trend.signals')
    ok &= _same(trend.strength(s_cut), trend.strength(s).loc[:cut], 'trend.strength')
    # 4. 相對強弱訊號
    import universe as U
    d = U.load()
    c = d['close'].iloc[:, :40]
    cc = c[c.index <= c.index[-300]]
    a, b = relstrength.signals(cc), relstrength.signals(c)
    for k in a:
        ok &= _same(a[k], b[k].loc[:cc.index[-1]], f'relstrength.signals[{k}]')
    # 4b. 合併面板特徵（不含答案欄）
    import panel
    cols = panel.pool(d, 'tw')[:60]
    full = {k: v[cols] for k, v in d.items()}
    cutd = full['close'].index[-400]
    trunc = {k: v[v.index <= cutd] for k, v in full.items()}
    Pa, _ = panel.make_panel(trunc, cols)
    Pb, _ = panel.make_panel(full, cols)
    Pb = Pb[Pb.index.get_level_values('date') <= cutd]
    ok &= _same(Pa[panel.FEATS].dropna(how='all'), Pb[panel.FEATS].dropna(how='all'), 'panel.make_panel 特徵（截斷 = 完整）')
    # 5. 竄改未來價格：過去的狀態不變
    s2 = s.copy()
    s2[s2.index > cut] *= np.random.default_rng(0).uniform(0.5, 1.5, (s2.index > cut).sum())
    ok &= _same(regime.walk_forward(s2).loc[:cut], regime.walk_forward(s_cut), 'regime 竄改未來價格後，過去不變')
    print('全部通過' if ok else '有失敗項目')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
