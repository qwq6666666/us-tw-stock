"""三大法人買賣超（證交所 T86，上市個股逐日資料）：回補歷史、每日補新、載入成模型特徵。

- 每個交易日一個請求：https://www.twse.com.tw/rwd/zh/fund/T86?date=YYYYMMDD&selectType=ALL
- 只留關注清單內的台股（外資、投信、自營商、三大法人合計，單位：股），另存當日全市場合計（代號 ALL）。
- 資料存 data/institutional.csv；已處理過的日期（含休市日）記在 data/institutional_state.json，可中斷續跑。
- 請求間隔 3 秒，請不要調快，避免被證交所限流。
- 時間點規則：這份資料盤後（約 16:30）才公布，所以「交易日 t 的判斷」只能用到 t-1 以前的資料（見 model.py 的 shift）。
手動回補：python institutional.py --start 2023-01-01
"""
import argparse
import csv
import datetime as dt
import json
import os
import time

import pandas as pd
import requests

import core

CSV = os.path.join(core.DATA_DIR, 'institutional.csv')
STATE = os.path.join(core.DATA_DIR, 'institutional_state.json')
HEADERS = {'User-Agent': 'Mozilla/5.0'}
SLEEP = 3.0
COLS = ['date', 'code', 'foreign', 'trust', 'dealer', 'total']
WATCH = sorted({t.split('.')[0] for t in core.TW_COLS if t.endswith('.TW')})


def _n(x):
    try:
        return float(str(x).replace(',', '').strip())
    except ValueError:
        return 0.0


def fetch_day(day):
    """回傳該日的列（清單內個股＋ALL）；休市日回傳空串列；連線失敗拋出例外。"""
    r = None
    for attempt in range(5):
        try:
            r = requests.get('https://www.twse.com.tw/rwd/zh/fund/T86',
                             params={'date': f'{day:%Y%m%d}', 'selectType': 'ALL', 'response': 'json'},
                             headers=HEADERS, timeout=30).json()
            break
        except Exception:
            time.sleep(15 * (attempt + 1))        # 可能被限流，拉長等待
    if r is None:
        raise RuntimeError(f'{day:%Y-%m-%d} 多次連線失敗')
    if r.get('stat') != 'OK' or not r.get('data'):
        if '沒有符合條件' in str(r.get('stat', '')):
            return []                              # 休市日
        raise RuntimeError(f"{day:%Y-%m-%d} 證交所回應異常：{str(r.get('stat'))[:60]}")
    rows, tot = [], [0.0, 0.0, 0.0, 0.0]
    for x in r['data']:
        f = _n(x[4]) + _n(x[7])        # 外資 = 外陸資 + 外資自營商
        tr, de, al = _n(x[10]), _n(x[11]), _n(x[18])
        for i, v in enumerate((f, tr, de, al)):
            tot[i] += v
        code = x[0].strip()
        if code in WATCH:
            rows.append([f'{day:%Y-%m-%d}', code, f, tr, de, al])
    rows.append([f'{day:%Y-%m-%d}', 'ALL'] + tot)
    return rows


def _state():
    return set(json.load(open(STATE, encoding='utf-8'))) if os.path.exists(STATE) else set()


def fill(start, end, max_days=0, verbose=True):
    """補齊 [start, end] 內尚未處理的平日。回傳新增的列數。"""
    done = _state()
    os.makedirs(core.DATA_DIR, exist_ok=True)
    fresh = not os.path.exists(CSV)
    added, n = 0, 0
    with open(CSV, 'a', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        if fresh:
            w.writerow(COLS)
        day = start
        while day <= end:
            key = f'{day:%Y-%m-%d}'
            if day.weekday() < 5 and key not in done:
                rows = fetch_day(day)
                w.writerows(rows)
                f.flush()
                added += len(rows)
                done.add(key)
                json.dump(sorted(done), open(STATE, 'w', encoding='utf-8'))
                n += 1
                if verbose and n % 20 == 0:
                    print(f'{key}：已處理 {n} 天，新增 {added} 列', flush=True)
                time.sleep(SLEEP)
                if max_days and n >= max_days:
                    break
            day += dt.timedelta(days=1)
    return added


def load():
    """回傳 date × code 的長表；沒有檔案時回傳 None。"""
    if not os.path.exists(CSV):
        return None
    d = pd.read_csv(CSV, parse_dates=['date'], dtype={'code': str})
    return d.drop_duplicates(subset=['date', 'code'], keep='last')


def update_recent(days=10):
    """每日更新用：補最近幾天尚未處理的平日（證交所約 16:30 後才有當天資料，當天沒資料會留到明天補）。"""
    today = dt.date.today()
    done = _state()
    start = today - dt.timedelta(days=days)
    # 今天若還沒公布會被記成「休市」；為避免誤記，今天不處理，留到下次
    return fill(start, today - dt.timedelta(days=1), verbose=False) if start else 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2023-01-01')
    ap.add_argument('--max-days', type=int, default=0)
    a = ap.parse_args()
    n = fill(dt.date.fromisoformat(a.start), dt.date.today() - dt.timedelta(days=1), a.max_days)
    print('完成，新增', n, '列')
