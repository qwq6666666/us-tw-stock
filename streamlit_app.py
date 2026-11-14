"""美股 × 台股整合分析：網頁介面（Streamlit）。資料來自 data/ 資料夾，由每日排程更新。"""
import json
import os
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import core
import fundamentals
import model as probmodel

st.set_page_config(page_title='美股 × 台股 整合分析', page_icon='📈', layout='wide')


def _mtime(path):
    return os.path.getmtime(path) if os.path.exists(path) else 0


# 以檔案修改時間當快取鍵：每日排程存入新資料後，網頁會自動讀到最新內容
@st.cache_resource(show_spinner=False)
def get_market(mtime):
    return core.load_market()


@st.cache_data(show_spinner=False)
def get_news(mtime):
    return core.load_news()


@st.cache_data(ttl=1800, show_spinner='向 Yahoo Finance 查詢中…')
def fetch_live(ticker):
    df = core.download_close([ticker])
    return df[ticker].dropna() if ticker in df.columns else None


def show(fig):
    st.plotly_chart(fig, width='stretch')


def price_chart(s, title):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.72, 0.28], vertical_spacing=0.04)
    fig.add_trace(go.Scatter(x=s.index, y=s, name='收盤價'), row=1, col=1)
    for n, nm in ((20, '月線'), (60, '季線'), (240, '年線')):
        fig.add_trace(go.Scatter(x=s.index, y=s.rolling(n).mean(), name=nm, line=dict(width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=s.index, y=core.rsi(s), name='RSI', showlegend=False), row=2, col=1)
    fig.add_hline(y=70, line_dash='dot', line_color='gray', row=2, col=1)
    fig.add_hline(y=30, line_dash='dot', line_color='gray', row=2, col=1)
    fig.update_layout(title=title, height=520, hovermode='x unified', margin=dict(l=10, r=10),
                      legend=dict(orientation='h', y=1.08))
    fig.update_yaxes(range=[0, 100], title_text='RSI', row=2, col=1)
    return fig


def lookup(code, M):
    """回傳（名稱, 價格序列, 錯誤訊息）。清單內的用已存資料，其餘現場查詢。"""
    code = code.strip().upper()
    candidates = [code + '.TW', code + '.TWO'] if re.fullmatch(r'\d{4,6}[A-Z]?', code) else [code]
    for c in candidates:
        if c in M.close_all.columns:
            return core.label(c), M.close_all[c].dropna(), None
    err = ''
    for c in candidates:
        try:
            s = fetch_live(c)
            if s is not None and len(s) > 5:
                return c, s, None
        except Exception as e:
            err = f'（{type(e).__name__}）'
    return None, None, (f'查不到「{code}」的資料{err}。請確認代號：台股輸入數字即可，美股輸入英文代號。'
                        '若代號正確，可能是資料來源暫時限流，請稍後再試。')


META = json.load(open(core.META_JSON, encoding='utf-8')) if os.path.exists(core.META_JSON) else {}

st.title('美股 × 台股 整合分析')
st.caption(f"資料更新時間：{META.get('updated_at', '尚未更新')}（台北時間）　｜　僅供研究與學習，不構成投資建議")

if not os.path.exists(core.PRICES_CSV):
    st.warning('尚無資料。請到 GitHub 專案的 Actions 頁面手動執行一次「每日更新」。')
    st.stop()

M = get_market(_mtime(core.PRICES_CSV))
NEWS = get_news(_mtime(core.NEWS_CSV))
SUMMARY = open(core.SUMMARY_MD, encoding='utf-8').read() if os.path.exists(core.SUMMARY_MD) else ''

FUND = fundamentals.load()
PROB = probmodel.load()

tab_sum, tab_stock, tab_fund, tab_prob, tab_news, tab_perf, tab_link, tab_adr = st.tabs(
    ['今日摘要', '個股查詢', '公司體質', '漲跌機率', '新聞', '績效比較', '美股 → 台股連動', 'ADR 溢價'])

with tab_sum:
    st.markdown(SUMMARY, unsafe_allow_html=True)

with tab_stock:
    with st.form('lookup'):
        code = st.text_input('股票代號', placeholder='台股輸入數字（2330、0050），美股輸入英文（NVDA、TSLA）')
        go_btn = st.form_submit_button('查詢', type='primary')
    if go_btn and code.strip():
        name, s, err = lookup(code, M)
        if err:
            st.error(err)
        else:
            st.markdown(core.describe_stock(s, name))
            show(price_chart(s, name))

def fmt_metric(key, v):
    if v is None:
        return '—'
    return f'{v:.1%}' if fundamentals.METRICS[key][1] == 'pct' else f'{v:.2f} 倍'


with tab_fund:
    if not FUND or not (FUND.get('tw') or FUND.get('us')):
        st.info('尚無公司體質資料。請到 GitHub 專案的 Actions 頁面手動執行一次「每日更新」。')
    else:
        st.caption(f"更新：{FUND['updated_at']}　｜　台股：{FUND.get('tw_period', '—')}；美股：yfinance 最新一季與近四季　｜　"
                   '評分是用固定門檻換算的粗略尺度（0 分最差、100 分最好），僅供研究，不構成投資建議。')
        pool = {}
        for code, v in FUND.get('tw', {}).items():
            pool[f"{v['name']}（{code}）"] = v
        for code, v in FUND.get('us', {}).items():
            pool[f"{v['name']}（{code}）"] = v
        pick = st.selectbox('選擇個股（關注清單內）', list(pool))
        d = pool[pick]

        c1, c2 = st.columns([1, 2])
        c1.metric('體質綜合分數', '—' if d['total'] is None else f"{d['total']:.0f} / 100", d['grade'], delta_color='off')
        c2.markdown(d['verdict'])

        cats = [k for k in fundamentals.DIMENSIONS if d['scores'].get(k) is not None]
        vals = [d['scores'][k] for k in cats]
        left, right = st.columns(2)
        if cats:
            fig = go.Figure(go.Scatterpolar(r=vals + vals[:1], theta=cats + cats[:1], fill='toself', name=pick))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), height=360, showlegend=False,
                              margin=dict(l=30, r=30, t=30, b=30))
            left.plotly_chart(fig, width='stretch')
        with right:
            for dim in fundamentals.DIMENSIONS:
                sc = d['scores'].get(dim)
                st.markdown(f'**{dim}**　{"資料不足" if sc is None else f"{sc:.0f} 分"}')
                st.progress(0 if sc is None else int(sc))
                rows = [{'指標': fundamentals.METRICS[k][0], '數值': fmt_metric(k, d['metrics'].get(k)),
                         '意思': fundamentals.METRICS[k][2]}
                        for k, _, _ in fundamentals.SCORING[dim] if k in d['metrics']]
                if rows:
                    with st.expander('看計算依據'):
                        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

        st.subheader('同業比較')
        st.caption(f"比較對象：{d['peer_label']}。「同業排名」：100 = 同業中最好、0 = 最差（對該指標而言，如負債比、本益比是越低越好）。")
        prow = []
        for r in d['peers']:
            nm = fundamentals.METRICS[r['key']][0]
            prow.append({'指標': nm, '本檔': fmt_metric(r['key'], r['mine']), '同業中位數': fmt_metric(r['key'], r['median']),
                         '同業排名': '樣本太少' if r['rank'] is None else f"{r['rank']}（共 {r['n']} 家）"})
        st.dataframe(pd.DataFrame(prow), hide_index=True, width='stretch')
        with st.expander('評分怎麼算？限制是什麼？'):
            st.markdown('\n'.join([
                '- 每個指標依固定門檻換算成 0～100 分（例如營業利益率 0% 得 0 分、25% 以上得 100 分），'
                '面向分數是該面向指標的平均，綜合分數是五個面向的平均。',
                '- 門檻是通用尺度，**不分產業**：代工、零售本來毛利低，所以請搭配上面的「同業比較」看。',
                '- 「股東回報」只看現金殖利率，成長股常把錢拿去再投資或買回庫藏股，分數偏低不代表不好。',
                '- 估值越便宜分數越高，但高成長公司本益比本來就高；「成長強、估值貴」要留意預期落空的風險。',
                '- 台股損益是年初累計數，ROE 以累計淨利年化估算，季節性強的公司會失真；'
                '獲利年增率台股暫缺（證交所資料沒有去年同期）。',
                '- 美股 ADR 若財報幣別與股價不同，不計股價淨值比。美股同業只有關注清單內的幾檔，樣本少。',
                '- ETF（如 0050）與指數沒有公司財報，不在此列。',
            ]))

def trend_state(t):
    """趨勢多空：收盤價在季線之上，且季線比 20 個交易日前高，才算「多」。"""
    if t not in M.close_all.columns:
        return None
    s = M.close_all[t].dropna()
    ma = s.rolling(60).mean()
    if len(s) < 90:
        return None
    return bool(s.iloc[-1] > ma.iloc[-1] and ma.iloc[-1] > ma.iloc[-21])


def fund_total(t):
    if not FUND:
        return None
    d = FUND.get('us', {}).get(t) if not t.endswith('.TW') else FUND.get('tw', {}).get(t[:-3])
    return None if d is None else d['total']


with tab_prob:
    if not PROB or not PROB.get('predictions'):
        st.info('尚無模型結果。請到 GitHub 專案的 Actions 頁面手動執行一次「每日更新」。')
    else:
        st.caption(f"更新：{PROB['updated_at']}　｜　預測的是「未來 N 個交易日後收盤價高於今天」的機率，不是買賣建議。")
        val = PROB['validation']
        n_ok = sum(1 for mk in val.values() for r in mk.values() if r and r['beats_baseline'])
        n_all = sum(1 for mk in val.values() for r in mk.values() if r)
        if n_ok == 0:
            st.warning('**請先看這裡：** 這個模型在「按時間切分」的歷史驗證中，**沒有勝過**最簡單的基準（永遠猜漲／永遠猜歷史上漲比例）。'
                       '意思是：只靠價量、美股隔夜、匯率這些資料，目前還抓不到可靠的規律。下方機率只是貼近歷史上漲比例的參考，'
                       '不要拿來當進出依據。')
        else:
            st.info(f'{n_all} 組驗證中有 {n_ok} 組勝過基準；沒勝過的組別，機率請特別保守看待。')

        names = {core.label(t): t for t in PROB['predictions']}
        pick = st.selectbox('選擇標的', list(names), key='prob_pick')
        t = names[pick]
        preds = PROB['predictions'][t]
        cols = st.columns(len(preds))
        for col, (h, p) in zip(cols, sorted(preds.items(), key=lambda kv: int(kv[0]))):
            col.metric(f"未來 {PROB['horizons'][h]}（{h} 個交易日）上漲機率", f"{p['prob']:.0%}",
                       f"歷史上這段期間平均 {p['base']:.0%} 會漲", delta_color='off')
        any_p = next(iter(preds.values()))
        st.caption(f"資料截至 {any_p['as_of']} 收盤。台股預測只用嚴格早於該日的美股收盤價。")
        st.markdown('**這次機率主要受哪些因素影響**（以 1 週預測為例）')
        for f in (preds.get('5') or any_p)['factors']:
            st.markdown(f"- {f['name']}：{f['value']} → **{f['effect']}** 上漲機率")

        st.subheader('體質好壞 × 趨勢多空')
        st.caption('體質：「公司體質」分頁的綜合分數 55 分以上算好。趨勢：收盤價在季線（60 日均線）之上、且季線比 20 個交易日前高算「多」。'
                   '這是描述現況的對照表，不是買賣訊號。')
        cells = {(True, True): [], (True, False): [], (False, True): [], (False, False): []}
        nodata = []
        for tt in list(M.sel_us) + list(M.sel_tw) + [x for x in M.us_cols + M.tw_cols if x not in M.sel_us + M.sel_tw]:
            if tt.startswith('^'):
                continue
            ft, tr = fund_total(tt), trend_state(tt)
            if ft is None or tr is None:
                nodata.append(core.label(tt))
            else:
                cells[(ft >= 55, tr)].append(f'{core.label(tt)} {ft:.0f} 分')
        desc = {(True, True): ('體質好 ＋ 趨勢多', '基本面與股價走勢都偏正面。'),
                (True, False): ('體質好 ＋ 趨勢空', '公司體質不差但股價走弱，可能是市場有疑慮或正在修正，值得查原因。'),
                (False, True): ('體質普通／弱 ＋ 趨勢多', '股價強但體質普通，可能靠題材或資金推動，波動風險較高。'),
                (False, False): ('體質普通／弱 ＋ 趨勢空', '基本面與股價走勢都偏弱。')}
        for row in ((True, True), (True, False)), ((False, True), (False, False)):
            for col, key in zip(st.columns(2), row):
                with col.container(border=True):
                    st.markdown(f'**{desc[key][0]}**')
                    st.caption(desc[key][1])
                    st.markdown('、'.join(cells[key]) if cells[key] else '（目前沒有）')
        if nodata:
            st.caption('無體質或趨勢資料（ETF、資料不足）：' + '、'.join(nodata))

        st.subheader('模型成績（按時間切分的歷史驗證）')
        st.caption('做法：只用某個時間點「以前」的資料訓練，預測「之後」半年，再往後滾動；訓練與驗證之間隔開預測天數，避免偷看未來。'
                   'AUC 0.5 代表跟亂猜一樣、越高越好；Brier 分數是機率誤差，越低越好，要低於「基準」才算有用。')
        rows = []
        for mk, mname in (('tw', '台股'), ('us', '美股')):
            for h, r in sorted(val.get(mk, {}).items(), key=lambda kv: int(kv[0])):
                if not r:
                    continue
                rows.append({'市場': mname, '期間': PROB['horizons'][h], '驗證樣本': r['n'], 'AUC': f"{r['auc']:.3f}",
                             '模型命中率': f"{r['acc']:.1%}", '永遠猜漲命中率': f"{r['acc_always_up']:.1%}",
                             'Brier（模型）': f"{r['brier']:.4f}", 'Brier（基準）': f"{r['brier_base']:.4f}",
                             '結論': '勝過基準' if r['beats_baseline'] else '沒勝過基準'})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
        with st.expander('怎麼讀這張表？為什麼可能沒勝過基準？'):
            st.markdown(chr(10).join([
                '- 這幾年股市大多在漲，所以「永遠猜漲」本身就有 56%～65% 的命中率，是很難超越的基準。',
                '- 只用價量、美股隔夜、匯率，資訊量有限；短期漲跌多半是隨機的，模型沒勝過基準是常見且誠實的結果。',
                '- 樣本只有約 5 年，且相鄰日期的結果高度重疊，真正獨立的樣本比表上的數字少很多。',
                '- 還沒納入法人買賣超（證交所沒有歷史資料，需要先每天累積）與新聞情緒，這是後續改進的方向。',
                '- 模型採保守設定（淺樹、強正則化），並試過三種複雜度，皆未勝過基準；沒有再針對驗證期反覆調參。']))

with tab_news:
    st.caption(f"累積 {len(NEWS)} 則　｜　" + '　｜　'.join(META.get('news_status', [])))
    c1, c2 = st.columns([1, 2])
    market = c1.radio('市場', ['全部', '台股', '美股'], horizontal=True)
    kw = c2.text_input('關鍵字或代號', placeholder='例如 台積電、NVDA')
    df = NEWS if market == '全部' else NEWS[NEWS['market'] == market]
    if kw.strip():
        k = kw.strip()
        df = df[df['title'].str.contains(k, case=False, regex=False)
                | df['ticker'].str.contains(k, case=False, regex=False)]
    view = pd.DataFrame({'時間': df['time'].dt.strftime('%m-%d %H:%M'), '來源': df['source'],
                         '標題': df['title'], '原文': df['url'], '相關代號': df['ticker']}).head(200)
    st.dataframe(view, hide_index=True, width='stretch', height=600,
                 column_config={'原文': st.column_config.LinkColumn(display_text='開啟', width='small'),
                                '標題': st.column_config.TextColumn(width='large')})

with tab_perf:
    options = M.sel_us + M.sel_tw
    c1, c2 = st.columns([3, 1])
    picks = c1.multiselect('標的', options, default=options, format_func=core.label)
    currency = c2.radio('計價幣別', ['新台幣', '美元', '各自原幣'])
    if picks:
        px_df = M.normalized(picks, currency)
        nm = px_df / px_df.iloc[0] * 100
        fig = go.Figure()
        for t in nm.columns:
            fig.add_trace(go.Scatter(x=nm.index, y=nm[t], name=core.label(t),
                                     line=dict(dash='solid' if M.is_us(t) else 'dot')))
        fig.update_layout(title=f'累積績效（起點 = 100，{currency}；實線美股、虛線台股）', yaxis_type='log',
                          hovermode='x unified', height=480, margin=dict(l=10, r=10))
        show(fig)
        stats = core.perf_stats(px_df).sort_values('夏普值', ascending=False)
        for c in ['年化報酬', '年化波動', '最大回撤', '期間總報酬']:
            stats[c] = stats[c].map('{:.1%}'.format)
        stats['夏普值'] = stats['夏普值'].map('{:.2f}'.format)
        st.dataframe(stats, width='stretch')
        if len(picks) >= 2:
            corr = px_df.resample('W-FRI').last().pct_change().dropna(how='all').corr()
            corr.index = corr.columns = [core.NAMES.get(t, t) for t in corr.columns]
            fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                            aspect='auto', title='週報酬相關係數（越接近 1，越常一起漲跌）', height=480)
            fig.update_layout(margin=dict(l=10, r=10))
            show(fig)

with tab_link:
    c1, c2 = st.columns(2)
    us = c1.selectbox('美股指標', M.sel_us, index=M.sel_us.index(core.US_LEAD) if core.US_LEAD in M.sel_us else 0,
                      format_func=core.label)
    tw = c2.selectbox('台股標的', M.sel_tw, index=M.sel_tw.index(core.TW_TARGET) if core.TW_TARGET in M.sel_tw else 0,
                      format_func=core.label)
    pair, bucket = M.leadlag(us, tw)
    k, b = np.polyfit(pair['us'], pair['tw'], 1)
    st.markdown(f'樣本 {len(pair)} 天，相關係數 **{pair.us.corr(pair.tw):.2f}**；美股每漲 1%，台股隔日平均變動 '
                f'**{k:.2f}%**；方向一致比例 **{(np.sign(pair.us) == np.sign(pair.tw)).mean():.0%}**')
    fig = px.scatter(pair, x='us', y='tw', opacity=0.4, height=440,
                     labels={'us': '美股隔夜報酬', 'tw': '台股隔日報酬'})
    xs = np.array([pair.us.min(), pair.us.max()])
    fig.add_trace(go.Scatter(x=xs, y=b + k * xs, mode='lines', name='迴歸線'))
    fig.update_layout(xaxis_tickformat='.0%', yaxis_tickformat='.0%', margin=dict(l=10, r=10))
    show(fig)
    st.dataframe(pd.DataFrame({'美股隔夜': bucket.index.astype(str), '天數': bucket['天數'].values,
                               '台股平均報酬': bucket['台股平均報酬'].map('{:+.2%}'.format).values,
                               '台股上漲機率': bucket['台股上漲機率'].map('{:.0%}'.format).values}),
                 hide_index=True, width='stretch')
    st.caption('台股的隔日報酬是收盤對收盤，其中大部分通常在開盤跳空時就已反映。')

with tab_adr:
    prem = M.adr_premium()
    fig = px.line(prem, height=460, title='ADR 相對台股的溢價（正）／折價（負）')
    fig.add_hline(y=0, line_dash='dash', line_color='gray')
    fig.update_layout(yaxis_tickformat='.0%', hovermode='x unified', legend_title='', margin=dict(l=10, r=10))
    show(fig)
    st.dataframe(pd.DataFrame({'最新': prem.ffill().iloc[-1], '近一年平均': prem.tail(core.TRADING_DAYS).mean(),
                               '期間最高': prem.max(), '期間最低': prem.min()}).map('{:+.1%}'.format),
                 width='stretch')
