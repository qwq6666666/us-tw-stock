"""每日更新：下載行情、蒐集新聞、產生今日摘要，全部存進 data/ 資料夾。"""
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import core
import fundamentals
import institutional
import model
import outlook
import scorecard
import trend
import volmodel


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

    close, unfinished = core.drop_unfinished(close)      # 盤中執行時，不採用尚未收盤的當日價格
    missing = [t for t in tickers if t not in close.columns]
    close.to_csv(core.PRICES_CSV, encoding='utf-8-sig')
    market = core.Market(close)

    news, status = core.fetch_all_news()
    if unfinished:
        status.append('已略過尚未收盤的價格：' + '、'.join(unfinished))
    archive = core.update_archive(news)

    try:                                          # 公司體質：失敗時保留前一次的檔案，不影響其他更新
        fund = fundamentals.build_all()
        if fund['tw'] or fund['us']:
            fundamentals.save(fund)
        status += ['公司體質：' + x for x in fund['status']]
    except Exception as e:
        status.append(f'公司體質：失敗（{type(e).__name__}: {str(e)[:80]}）')

    try:                                          # 漲跌機率模型：失敗時保留前一次的檔案
        try:                                          # 法人買賣超：補最近幾天；失敗不影響模型（沒有就不用）
            n_inst = institutional.update_recent()
            status.append(f'法人買賣超：新增 {n_inst} 列')
        except Exception as e:
            status.append(f'法人買賣超：失敗（{type(e).__name__}: {str(e)[:80]}）')
        mres = model.build_all(market, institutional.load())
        model.save(mres)
        added = scorecard.append_predictions(mres)          # 只新增、不修改；同一天重跑不會重複記
        status.append(f'模型成績單：新增 {added} 筆判斷紀錄')
        resolved = scorecard.resolve_new(market.close_all)    # 剛到期的結果寫進只能新增的結果紀錄
        status.append(f'模型成績單：新增 {resolved} 筆到期結果')
        status += ['漲跌機率模型：' + x for x in mres['status']]
    except Exception as e:
        status.append(f'漲跌機率模型：失敗（{type(e).__name__}: {str(e)[:80]}）')

    try:                                          # 趨勢規則實測與趨勢強度；失敗時保留前一次的檔案
        tres = trend.build_all(market)
        trend.save(tres)
        status += ['趨勢實測：' + x for x in tres['status']]
    except Exception as e:
        status.append(f'趨勢實測：失敗（{type(e).__name__}: {str(e)[:80]}）')

    try:                                          # 波動預測：模型、只能新增的預測紀錄、到期結果；失敗時保留前一次的檔案
        vres = volmodel.build_all(market)
        volmodel.save(vres)
        v_new, v_done = volmodel.append_predictions(vres), volmodel.resolve_new(market.close_all)
        status += ['波動預測：' + x for x in vres['status']]
        status.append(f'波動成績單：新增 {v_new} 筆預測、{v_done} 筆到期結果')
    except Exception as e:
        status.append(f'波動預測：失敗（{type(e).__name__}: {str(e)[:80]}）')

    try:                                          # 新聞前景模擬：用歷史新聞做不偷看未來的滾動模擬；失敗時保留前一次的檔案
        ores = outlook.build_all(market.close_all)
        if ores.get('summary'):
            outlook.save(ores)
        status.append(f"新聞前景模擬：{ores['n_news']} 則新聞、{ores.get('n_samples', 0)} 筆樣本")
    except Exception as e:
        status.append(f'新聞前景模擬：失敗（{type(e).__name__}: {str(e)[:80]}）')

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
