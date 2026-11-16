"""只能新增的紀錄檔（CSV）＋ SHA-256 雜湊鏈：每列的 hash 含前一列的 hash，事後改動或刪除任何一列都會讓驗證失敗。

給新的紀錄使用（例如波動預測）。舊的 scorecard.py 有自己的實作，格式已定，不要改用這裡的函式重寫。
"""
import csv
import hashlib
import os

GENESIS = '0' * 64


def read(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def _hash(prev, row, fields):
    body = '|'.join(str(row[k]) for k in fields)
    return hashlib.sha256((prev + '|' + body).encode('utf-8')).hexdigest()


def append(path, fields, rows, key):
    """把 rows（dict，值須已是字串格式）加到檔尾。key 欄位組合已存在的列會略過（只記第一次）。回傳新增筆數。"""
    old = read(path)
    seen = {tuple(r[k] for k in key) for r in old}
    prev = old[-1]['hash'] if old else GENESIS
    new = []
    for row in rows:
        k = tuple(str(row[x]) for x in key)
        if k in seen:
            continue
        seen.add(k)
        out = {x: str(row[x]) for x in fields}
        out['prev_hash'] = prev
        out['hash'] = prev = _hash(prev, out, fields)
        new.append(out)
    if new:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fresh = not os.path.exists(path)
        with open(path, 'a', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(fields) + ['prev_hash', 'hash'], lineterminator='\n')
            if fresh:
                w.writeheader()
            w.writerows(new)
    return len(new)


def verify(path, fields):
    """回傳（是否完整, 筆數, 第一個出問題的列號或 None）。"""
    prev = GENESIS
    rows = read(path)
    for i, r in enumerate(rows, 1):
        if r['prev_hash'] != prev or _hash(prev, r, fields) != r['hash']:
            return False, len(rows), i
        prev = r['hash']
    return True, len(rows), None
