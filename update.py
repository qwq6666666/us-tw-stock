"""每日更新：下載行情、蒐集新聞、產生今日摘要，全部存進 data/ 資料夾。"""
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import core
import fundamentals
import model
import scorecard


def main():
    os.makedirs(core.DATA_DIR, exist_ok=True)
    tickers = core.US_COLS + core.TW_COLS + [core.FX_TICKER]

    close = None
    for attempt in range(3):                      # Yahoo 偶爾會暫時限流，失敗就稍等再試
        try:
            close = core.download_close(tickers)
            if core.FX_TICKER in close.columns and len(close) > 300:
                break
        except Exception as e:
            print(f'第 {attempt + 1} 次下載失敗：{e}')
        close = None
        time.sleep(20 * (attempt + 1))
    if close is None:
        print('行情下載失敗，保留前一次的資料。')
        sys.exit(1)

    missing = [t for t in tickers if t not in close.columns]
    close.to_csv(core.PRICES_CSV, encoding='utf-8-sig')
    market = core.Market(close)

    news, status = core.fetch_all_news()
    archive = core.update_archive(news)

    try:                                          # 公司體質：失敗時保留前一次的檔案，不影響其他更新
        fund = fundamentals.build_all()
        if fund['tw'] or fund['us']:
            fundamentals.save(fund)
        status += ['公司體質：' + x for x in fund['status']]
    except Exception as e:
        status.append(f'公司體質：失敗（{type(e).__name__}: {str(e)[:80]}）')

    try:                                          # 漲跌機率模型：失敗時保留前一次的檔案
        mres = model.build_all(market)
        model.save(mres)
        added = scorecard.append_predictions(mres)          # 只新增、不修改；同一天重跑不會重複記
        status.append(f'模型成績單：新增 {added} 筆判斷紀錄')
        status += ['漲跌機率模型：' + x for x in mres['status']]
    except Exception as e:
        status.append(f'漲跌機率模型：失敗（{type(e).__name__}: {str(e)[:80]}）')

    with open(core.SUMMARY_MD, 'w', encoding='utf-8') as f:
        f.write(core.build_summary(market))

    now = datetime.now(timezone(timedelta(hours=8)))
    meta = {'updated_at': now.strftime('%Y-%m-%d %H:%M'), 'missing_tickers': missing,
            'news_status': status, 'news_new': len(news), 'news_total': len(archive)}
    with open(core.META_JSON, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"更新完成 {meta['updated_at']}（台北時間）")
    if missing:
        print('沒有抓到資料的代號：', missing)
    print('\n'.join(status))
    print(f'新聞本次 {len(news)} 則，累積 {len(archive)} 則')


if __name__ == '__main__':
    main()
