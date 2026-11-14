"""一次性工具：回補鉅亨網歷史新聞（只留與關注清單相關的），存成 data/news_history.csv。

用途：讓「新聞 → 前景判斷」模型能用過去幾年的新聞做「不偷看未來」的模擬（見 outlook.py）。
- 不放進每日排程；手動執行：python backfill_news.py --start 2023-01-01
- 可以中斷再續跑（進度記在 data/news_history_state.json；已抓過的週不會重抓）。
- 以每週為單位請求，每次請求間隔 0.6 秒，不要調得更快，避免對對方網站造成負擔。
- 相關判斷：新聞的股票標籤含關注清單的代號，或標題／摘要提到清單內公司名稱。
"""
import argparse
import csv
import datetime as dt
import json
import os
import time

import requests

import core

OUT = os.path.join(core.DATA_DIR, 'news_history.csv')
STATE = os.path.join(core.DATA_DIR, 'news_history_state.json')
HEADERS = {'User-Agent': 'Mozilla/5.0'}
TZ = dt.timezone(dt.timedelta(hours=8))
SLEEP = 0.6
CATEGORIES = {'tw_stock': '台股', 'us_stock': '美股'}

# 代號 -> 名稱關鍵字（用來補抓沒打標籤的新聞）
NAME_KEYS = {
    '2330': ['台積電'], '2317': ['鴻海'], '2454': ['聯發科'], '2303': ['聯電'], '3711': ['日月光'],
    '2412': ['中華電'], '0050': ['元大台灣50'], 'NVDA': ['輝達', 'NVIDIA'], 'AAPL': ['蘋果公司', 'Apple'],
    'TSM': ['台積電ADR'], 'UMC': [], 'ASX': [], 'CHT': [],
}
WATCH = set()
for t in core.US_COLS + core.TW_COLS:
    if not t.startswith('^'):
        WATCH.add(t.split('.')[0])


def match(item):
    """回傳這則新聞相關的清單代號（逗號分隔）；不相關回傳空字串。"""
    hit = set()
    for m in item.get('market') or []:
        code = str(m.get('code', ''))
        if code in WATCH:
            hit.add(code)
    text = (item.get('title') or '') + (item.get('summary') or '')
    for code, keys in NAME_KEYS.items():
        if code in WATCH and any(k in text for k in keys):
            hit.add(code)
    return ','.join(sorted(hit))


def fetch_window(cat, start, end):
    page, rows, seen = 1, [], 0
    while True:
        for attempt in range(4):
            try:
                r = requests.get(f'https://api.cnyes.com/media/api/v1/newslist/category/{cat}',
                                 params={'limit': 30, 'page': page, 'startAt': int(start.timestamp()), 'endAt': int(end.timestamp())},
                                 headers=HEADERS, timeout=20)
                r.raise_for_status()
                j = r.json()['items']
                break
            except Exception:
                time.sleep(3 * (attempt + 1))
        else:
            if page > 1:     # 已抓到前幾頁、後面的頁碼失敗（常見於分頁上限）：保留已抓到的並記錄，不讓整個回補中斷
                print(f'警告：{cat} {start:%Y-%m-%d} 第 {page} 頁起無法取得，該週只收錄前 {page - 1} 頁', flush=True)
                return rows, seen
            raise RuntimeError(f'{cat} {start:%Y-%m-%d} 第 {page} 頁多次失敗')
        for it in j['data']:
            seen += 1
            codes = match(it)
            if not codes:
                continue
            ts = dt.datetime.fromtimestamp(int(it['publishAt']), tz=TZ).replace(tzinfo=None)
            rows.append([ts.strftime('%Y-%m-%d %H:%M:%S'), CATEGORIES[cat], '鉅亨網', codes, it.get('title', ''),
                         (it.get('summary') or '').replace('\n', ' '), f"https://news.cnyes.com/news/id/{it.get('newsId', '')}"])
        if page >= (j.get('last_page') or 1):
            return rows, seen
        page += 1
        time.sleep(SLEEP)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2023-01-01')
    ap.add_argument('--end', default=None, help='預設到今天')
    ap.add_argument('--max-weeks', type=int, default=0, help='只做前 N 週（測試用）')
    a = ap.parse_args()
    start = dt.datetime.fromisoformat(a.start).replace(tzinfo=TZ)
    end = (dt.datetime.fromisoformat(a.end) if a.end else dt.datetime.now()).replace(tzinfo=TZ)
    done = set(json.load(open(STATE, encoding='utf-8'))) if os.path.exists(STATE) else set()
    os.makedirs(core.DATA_DIR, exist_ok=True)
    fresh = not os.path.exists(OUT)
    n_weeks = 0
    with open(OUT, 'a', encoding='utf-8-sig' if fresh else 'utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        if fresh:
            w.writerow(core.NEWS_COLS)
        cur = start
        while cur < end:
            nxt = min(cur + dt.timedelta(days=7), end)
            for cat in CATEGORIES:
                key = f'{cat}|{cur:%Y-%m-%d}'
                if key in done:
                    continue
                rows, seen = fetch_window(cat, cur, nxt)
                w.writerows(rows)
                f.flush()
                done.add(key)
                json.dump(sorted(done), open(STATE, 'w', encoding='utf-8'))
                print(f'{key}：掃 {seen} 則，留 {len(rows)} 則', flush=True)
                time.sleep(SLEEP)
            cur = nxt
            n_weeks += 1
            if a.max_weeks and n_weeks >= a.max_weeks:
                break
    print('完成')


if __name__ == '__main__':
    main()
